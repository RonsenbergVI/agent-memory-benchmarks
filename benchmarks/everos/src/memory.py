# MIT License
#
# Copyright (c) 2026 René-Jean Corneille
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""EverOS (EverMind-AI/EverOS) — local-first Markdown-native memory.

Workspace package ``everos-benchmark`` (this directory). EverOS runs
in-process: memories are Markdown files under ``EVEROS_ROOT`` with a
SQLite index beside them, so there is no server for compose to stand up.

Ingestion is EverOS's own. ``memorize`` takes a session's messages and
its pipeline extracts episodes, atomic facts and a user profile from
them, using the LLM and embedder EverOS is configured with. That
configuration is entirely environment-bound (``EVEROS_LLM__*``,
``EVEROS_EMBEDDING__*``), so the adapter points it at OpenAI over the
compatible protocol EverOS speaks natively. The spend is in-process
through the openai SDK, so ``OpenAIUsageTracker`` sees it — this
system's cost is measured, not assumed.

gpt-5-mini needs one accommodation to work at all here: everalgo asks
for ``temperature=0.0`` on every call and OpenAI's reasoning models
reject anything but the default, so ``_pin_temperature`` lifts the
configured temperature for those models. It is the same accommodation
mem0 carries for the same models and the same 400, and the narrowest
one available — everalgo reads ``LLMConfig.temperature``, so nothing
internal is patched. ``--param reasoning=false`` opts out.

Two properties shape this adapter and are not obvious:

* **A write does not necessarily extract.** ``memorize`` accumulates
  messages and only runs the extraction pipeline when its boundary
  detector says a topic ended — the return says ``accumulated`` or
  ``extracted``. Ingesting a session and querying immediately would
  therefore search a store that still holds the session as unprocessed
  messages. Each dataset session is flushed (``is_final=True``) once its
  turns are in, which is both correct and faithful: a dataset session
  *is* a conversation boundary.
* **Its API is async and its locks are module-level.** ``memorize``
  serialises per session on an ``asyncio.Lock`` held in module state, and
  a lock binds to the first event loop that contends on it — so one loop
  per worker thread raises "bound to a different event loop". Everything
  is marshalled onto ONE process-wide loop through ``_await``.

Each conversation is an EverOS *project*, which is both the search filter
and a directory segment under the memory root, so recall cannot cross
conversations and teardown is a directory removal scoped to one id.
"""

import asyncio
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
# EverOS's own default retrieval: "hybrid" is vector + keyword. "agentic"
# adds an LLM planning pass, which is query understanding rather than
# retrieval and would not be comparable with the other systems.
DEFAULT_SEARCH_METHOD = "hybrid"
# relative, so it lands under the container's WORKDIR; EVEROS_ROOT moves it
DEFAULT_ROOT = ".everos"
# The one temperature OpenAI's reasoning models accept. everalgo asks for
# 0.0 and EverOS builds its LLMConfig from only model/api_key/base_url,
# so there is no configuration path to change it and gpt-5-mini answers
# every call with a 400. `reasoning` lifts the configured temperature to
# this instead — the same accommodation mem0 carries for the same models
# and the same 400, and the narrowest one that exists: everalgo reads
# `LLMConfig.temperature`, so nothing internal is patched.
REASONING_TEMPERATURE = 1.0
# model families that reject any temperature but the default
_REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")
# the app every conversation of this benchmark is filed under
APP_ID = "amb"
# app_id / project_id / sender_id become directory segments, so EverOS
# admits only this charset and rejects "." and ".."
_PATH_SAFE = re.compile(r"[^a-zA-Z0-9_.@+-]+")

# One process-wide loop, running in its own thread; see the module
# docstring for why per-thread loops cannot work here.
_LOOP: asyncio.AbstractEventLoop | None = None
_GUARD = threading.Lock()
# EverOS's runtime is brought up once per process, serially: see
# `_start_runtime`. The lock is held across the await, so a worker
# arriving mid-startup waits for it rather than racing past into a
# half-built runtime.
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_READY = False
# The cascade provider, kept from startup so a write can drive its index
# synchronously; see `_sync_index`.
_CASCADE: Any = None


def _loop() -> asyncio.AbstractEventLoop:
    """The shared loop, started on first use."""
    global _LOOP
    with _GUARD:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_LOOP.run_forever, name="everos-loop", daemon=True
            ).start()
    return _LOOP


def _await(coro: Any, timeout: float = 1800) -> Any:
    """Run a coroutine on the shared loop and wait for it."""
    return asyncio.run_coroutine_threadsafe(coro, _loop()).result(timeout)


class EverOSMemory(Memory):
    """EverOS: Markdown-native local memory with an extraction pipeline."""

    name: ClassVar[str] = "everos"
    description: ClassVar[str] = "EverOS — local-first Markdown-native memory"
    sdk_dist: ClassVar[str | None] = "everos"

    def __init__(
        self,
        root: str | None = None,
        model: str | None = DEFAULT_INGESTION_MODEL,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int | str = DEFAULT_EMBEDDING_DIMENSIONS,
        search_method: str = DEFAULT_SEARCH_METHOD,
        reasoning: bool | str | None = None,
        **params: object,
    ) -> None:
        """Pin the models and the retrieval method EverOS will use."""
        super().__init__(**params)
        self.root = Path(root or os.environ.get("EVEROS_ROOT", DEFAULT_ROOT))
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings
        self.embedding_dimensions = int(embedding_dimensions)
        self.search_method = search_method
        # None auto-detects from the model name; --param reasoning=false
        # restores everalgo's own temperature for a model that accepts it
        if isinstance(reasoning, str):
            reasoning = reasoning.strip().lower() not in ("false", "0", "no")
        self.reasoning = (
            self._is_reasoning_model(self.model) if reasoning is None else reasoning
        )
        # the conversation this instance was built for; teardown needs it
        # and the base contract does not pass it
        self._conversation_id: str | None = None
        self._sessions: set[str] = set()
        self._extracted = 0

    @staticmethod
    def _is_reasoning_model(model: str | None) -> bool:
        """Whether this model rejects every temperature but the default."""
        return bool(model) and str(model).startswith(_REASONING_MODELS)

    def _pin_temperature(self) -> None:
        """Lift everalgo's configured temperature for a reasoning model.

        everalgo asks for 0.0 on every call and EverOS gives its
        ``LLMConfig`` only model/api_key/base_url, so a reasoning model
        answers with `'temperature' does not support 0.0`. The provider
        reads ``LLMConfig.temperature``, so changing that field's default
        is enough — no everalgo internal is patched, and a non-reasoning
        model is left exactly as EverOS shipped it.

        Must run before the LLM client is built, which the runtime's own
        ``LLMLifespanProvider`` does at startup.

        Raises:
            RuntimeError: if everalgo no longer exposes the field this
                accommodation sets.
        """
        if not self.reasoning:
            return
        from everalgo.llm.config import LLMConfig

        field = LLMConfig.model_fields.get("temperature")
        if field is None:  # everalgo moved it; fail loudly rather than 400
            raise RuntimeError(
                "everalgo LLMConfig has no `temperature` field; the reasoning-model "
                "accommodation in this adapter needs updating (--param reasoning=false "
                "to skip it)"
            )
        field.default = REASONING_TEMPERATURE
        LLMConfig.model_rebuild(force=True)

    @staticmethod
    def _slug(value: str) -> str:
        """A path-safe id: EverOS makes these directory segments."""
        slug = _PATH_SAFE.sub("-", value).strip("-.")
        return slug or "unnamed"

    def _project_id(self, conversation_id: str) -> str:
        """The project this conversation's memories live under, and only this one's."""
        return f"conv-{self._slug(conversation_id)}"

    def _user_id(self, conversation_id: str) -> str:
        """The one EverOS user a conversation's memories belong to.

        EverOS keys user memory by the message's ``sender_id`` and makes
        a search name exactly one of ``user_id``/``agent_id``. A LoCoMo
        conversation has two speakers, but the unit of memory here is the
        conversation, so both speak as one user and the speaker survives
        in ``sender_name`` and in the text of every message.
        """
        return f"conv-{self._slug(conversation_id)}"

    def _configure_env(self) -> None:
        """Set EverOS's environment before anything imports it.

        Its settings are read through a cached loader, and the section
        names bind as ``EVEROS_<SECTION>__<KEY>``. `setdefault` throughout
        so an operator's own configuration still wins.
        """
        os.environ.setdefault("EVEROS_ROOT", str(self.root.absolute()))
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if self.model:
            os.environ.setdefault("EVEROS_LLM__MODEL", self.model)
            os.environ.setdefault("EVEROS_LLM__API_KEY", key)
            os.environ.setdefault("EVEROS_LLM__BASE_URL", base_url)
        if self.embedding_model:
            os.environ.setdefault("EVEROS_EMBEDDING__MODEL", self.embedding_model)
            os.environ.setdefault("EVEROS_EMBEDDING__API_KEY", key)
            os.environ.setdefault("EVEROS_EMBEDDING__BASE_URL", base_url)
            os.environ.setdefault(
                "EVEROS_EMBEDDING__DIMENSIONS", str(self.embedding_dimensions)
            )

    def setup(self) -> None:
        """Configure EverOS, create its memory root, and start its runtime."""
        self._configure_env()
        self.root.mkdir(parents=True, exist_ok=True)
        self._scaffold_config()
        self._pin_temperature()
        self._start_runtime()

    def _scaffold_config(self) -> None:
        """Put EverOS's own default config files in the memory root.

        `everos init` copies two shipped templates into the root, and the
        OME engine refuses to start without `ome.toml`. Copying the same
        templates keeps the scaffold EverOS's rather than this adapter's;
        the models still come from the environment, which outranks the
        TOML.
        """
        from everos import config as everos_config

        templates = Path(everos_config.__file__).parent
        for name, template in (
            ("everos.toml", "default.toml"),
            ("ome.toml", "default_ome.toml"),
        ):
            target = self.root / name
            source = templates / template
            if not target.exists() and source.exists():
                shutil.copyfile(source, target)

    @staticmethod
    def _start_runtime() -> None:
        """Run EverOS's own startup, once per process, serially.

        EverOS brings its runtime up in the HTTP app's lifespan: the
        SQLite schema, the LanceDB store, the LLM and parser clients, the
        cascade, and the OME engine, in a defined order. Driving the
        service layer directly — which is what this adapter does, since
        there is no server — skips all of it, and the failures surface
        one at a time and late (`no such table: unprocessed_buffer`, then
        `emit: engine not started`).

        So rather than reimplementing the pieces, this runs EverOS's own
        providers in EverOS's own order against a throwaway app. Anything
        the project adds to that set comes along for free.
        """
        global _RUNTIME_READY
        with _RUNTIME_LOCK:
            if _RUNTIME_READY:
                return
            from everos.core.lifespan import (
                MetricsLifespanProvider,
                TracingLifespanProvider,
            )
            from everos.entrypoints.api.lifespans import (
                CascadeLifespanProvider,
                LanceDBLifespanProvider,
                LLMLifespanProvider,
                OmeLifespanProvider,
                ParserLifespanProvider,
                SqliteLifespanProvider,
            )
            from fastapi import FastAPI

            cascade = CascadeLifespanProvider()
            providers = [
                TracingLifespanProvider(),
                MetricsLifespanProvider(),
                LLMLifespanProvider(),
                ParserLifespanProvider(),
                SqliteLifespanProvider(),
                LanceDBLifespanProvider(),
                cascade,
                OmeLifespanProvider(),
            ]

            async def _start() -> None:
                app = FastAPI()
                for provider in sorted(providers, key=lambda p: p.order):
                    await provider.startup(app)

            _await(_start())
            global _CASCADE
            _CASCADE = cascade
            _RUNTIME_READY = True

    @staticmethod
    def _sync_index() -> None:
        """Drive the cascade once, so what was just written is searchable.

        Extraction writes Markdown and records a pending change; the
        cascade is what turns that into the LanceDB rows search reads,
        and it normally runs on a filesystem watcher and a schedule.
        A benchmark ingests and queries immediately, so the two are
        driven here instead of waited on — `sync_once` picks the change
        up and `drain_once` works the queue off before the next call
        returns.
        """
        orchestrator = getattr(_CASCADE, "_orchestrator", None)
        if orchestrator is None:
            return

        async def _sync() -> None:
            await orchestrator.sync_once()
            await orchestrator.drain_once()

        _await(_sync())

    def memorize(
        self,
        conversation_id: str,
        messages: list[dict],
        *,
        session_id: str,
        flush: bool = True,
    ) -> str:
        """Hand EverOS a batch of messages, then close the boundary.

        The flush is what turns accumulated messages into extracted
        memories; without it a session can sit unprocessed and be
        invisible to search. It is skipped only when the caller intends
        to keep adding to the same session.
        """
        from everos.service.memorize import memorize as everos_memorize

        self._conversation_id = conversation_id
        project_id = self._project_id(conversation_id)
        session_key = self._slug(session_id)
        payload = {
            "session_id": session_key,
            "app_id": APP_ID,
            "project_id": project_id,
            "messages": messages,
        }
        result = _await(everos_memorize(payload))
        status = getattr(result, "status", "")
        if flush:
            flushed = _await(
                everos_memorize(
                    {
                        "session_id": session_key,
                        "app_id": APP_ID,
                        "project_id": project_id,
                        "messages": [],
                    },
                    is_final=True,
                )
            )
            status = getattr(flushed, "status", status)
        self._sessions.add(session_key)
        if status == "extracted":
            self._extracted += 1
        self._sync_index()
        return status

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Feed the session's turns in, then close its boundary."""
        if not session.turns:
            return
        timestamp = self._epoch_ms(session.timestamp)
        messages = [
            {
                "sender_id": self._user_id(conversation_id),
                "sender_name": turn.speaker,
                "role": "user",
                # EverOS's contract is epoch milliseconds and rejects <= 0;
                # the ordering within a session is what matters, so turns
                # are spaced a second apart from the session's timestamp
                "timestamp": timestamp + index * 1000,
                "content": f"{turn.speaker}: {turn.text}",
            }
            for index, turn in enumerate(session.turns)
        ]
        self.memorize(conversation_id, messages, session_id=session.session_id)

    @staticmethod
    def _epoch_ms(timestamp: str | None) -> int:
        """The session's in-world time as epoch ms, or a safe stand-in.

        EverOS requires a strictly positive epoch-millisecond stamp, so a
        dataset without parseable timestamps still has to produce one.
        """
        from datetime import UTC, datetime

        if timestamp:
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%d %B %Y",
                "%I:%M %p on %d %B, %Y",
            ):
                try:
                    parsed = datetime.strptime(timestamp, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
                return int(parsed.timestamp() * 1000)
        # 2000-01-01, well clear of the > 0 floor and stable across runs
        return 946684800000

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k episodes from this conversation's project."""
        if not query.strip():
            return []
        from everos.memory.search.dto import SearchMethod, SearchRequest
        from everos.service.search import search as everos_search

        request = SearchRequest(
            user_id=self._user_id(conversation_id),
            app_id=APP_ID,
            project_id=self._project_id(conversation_id),
            query=query,
            method=SearchMethod(self.search_method),
            top_k=k,
        )
        try:
            response = _await(everos_search(request))
        except Exception:
            # the payload is worth having in the benchmark log, which
            # outlives the container
            logger.bind(scope="everos").error(
                "search failed: project={!r} method={!r} query={!r}",
                self._project_id(conversation_id),
                self.search_method,
                query,
            )
            raise
        hits = []
        for episode in response.data.episodes[:k]:
            hits.append(
                MemoryHit(
                    content=self._content(episode),
                    score=episode.score,
                    # session-level only: an episode is an extraction over
                    # a session's messages, not a verbatim turn
                    session_ids=[episode.session_id] if episode.session_id else [],
                    metadata={"episode_id": episode.id, "subject": episode.subject},
                )
            )
        return hits

    @staticmethod
    def _content(episode: Any) -> str:
        """The episode's text: its narrative, plus the facts drawn from it."""
        parts = [str(getattr(episode, "episode", "") or "").strip()]
        if not parts[0]:
            parts = [str(getattr(episode, "summary", "") or "").strip()]
        parts += [
            str(fact.content).strip()
            for fact in (getattr(episode, "atomic_facts", None) or [])
        ]
        return "\n".join(p for p in parts if p)

    def teardown(self) -> None:
        """Delete this conversation's project directory, and only this one's.

        Scoped to one project on purpose: the projects of the
        conversations running beside this one under `--workers N` are
        siblings under the same root, and a wider sweep would take them.
        """
        if self._conversation_id is not None:
            project = self.root / APP_ID / self._project_id(self._conversation_id)
            shutil.rmtree(project, ignore_errors=True)
        self._sessions.clear()

    def stats(self) -> dict:
        """Report what this run stored, and how it retrieved."""
        return {
            "sessions": len(self._sessions),
            "extracted_sessions": self._extracted,
            "search_method": self.search_method,
            "reasoning": self.reasoning,
        }
