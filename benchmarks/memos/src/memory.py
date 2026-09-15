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

"""MemOS (MemTensor/MemOS) — MemCube memory, scheduled OS-style.

Workspace package ``memos-benchmark`` (this directory). MemOS runs
in-process: a ``MOS`` instance over one ``GeneralMemCube`` whose textual
memory is the ``tree_text`` backend, which keeps its memories as a graph
in the Neo4j named by ``NEO4J_URI`` / ``NEO4J_USER`` / ``NEO4J_PASSWORD``.

The whole configuration comes from MemOS's own ``get_default_config`` /
``get_default_cube_config``, with only the models, the endpoints and the
ids injected — so what is measured is MemOS's own default stack rather
than this adapter's reading of it.

``tree_text`` is the backend on purpose: it is the one MemOS's own
``MOS.add`` runs its **MemReader** over, extracting memories from a
session's messages with the configured LLM. The other textual backends
take the same call and store each message verbatim, which would measure
a plain vector store wearing MemOS's name.

Three properties shape this adapter and are not obvious:

* **gpt-5-mini needs three parameters changed.** MemOS builds every
  chat request with ``temperature``, ``top_p`` and ``max_tokens``, and
  OpenAI's reasoning models reject all three (the last one by name, in
  favour of ``max_completion_tokens``). ``_reasoning_llm_class`` is a
  subclass of MemOS's own ``OpenAILLM`` that fixes the request body for
  those models and is registered as the ``openai`` backend; nothing else
  of MemOS is touched, and a non-reasoning model passes through
  unchanged. ``--param reasoning=false`` opts out.
* **MemOS's thread pools drop the ambient context.** Its
  ``ContextThreadPoolExecutor`` propagates only MemOS's own
  ``RequestContext``, and extraction, recall and embedding all run inside
  those pools. ``OpenAIUsageTracker`` keeps its counters in a
  ``ContextVar``, so every token spent in a MemOS worker thread would be
  booked into a fresh context and silently lost — the run would publish
  a confident zero it never earned. ``_propagate_context`` makes
  ``submit`` carry the submitting thread's context (``map`` goes through
  ``submit``, so it is covered too).
* **It writes its own diagnostics onto stdout**, which ``amb run``'s
  summary shares: its logger's console handler is configured there, and
  two of its graph methods ``print()`` the Cypher they run — including
  the query embedding, ~30 KB per search and per dedup lookup.
  ``_quiet_logging`` moves the logger to stderr (and its file handler
  off INFO, where the same dumps are hundreds of megabytes over a run),
  and ``_quiet_graph_dumps`` binds a ``print`` in the one module that
  makes those calls. Both are scoped to MemOS: no stream is replaced
  and no other library's output changes. Nothing is discarded either —
  ``--log-level debug`` prints all of it.

Each conversation is its own MemOS user and its own MemCube, and the
graph is shared with node-level isolation by ``user_name`` (Neo4j
Community has one database and no ``CREATE DATABASE``) — so recall
cannot cross conversations and teardown deletes one conversation's nodes
without touching the conversations running beside it under
``--workers N``.
"""

import contextvars
import logging
import os
import re
import sys
import threading
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
# MemOS's own default: vector + graph recall, reranked locally. "fine"
# adds an LLM pass over the query, which is query understanding rather
# than retrieval and would not be comparable with the other systems.
DEFAULT_SEARCH_MODE = "fast"
# MemOS's default config asks for 1024, which a reasoning model can spend
# entirely on reasoning tokens — the extraction then parses an empty
# response. 8192 is the ceiling MemOS's own LLMConfig defaults to.
DEFAULT_MAX_TOKENS = 8192
# model families that reject temperature, top_p and max_tokens
_REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")
# MemOS derives the graph's tenant tag from the user id by stripping "-"
# and "_", so ids that differ only in those would share a tenant; this
# charset keeps the derivation injective
_ID_SAFE = re.compile(r"[^a-z0-9]+")

# Held across MOS construction: the user store is one SQLite file shared
# by every instance in the process, and the graph's index creation is
# check-then-create. Workers arriving together would race both.
_SETUP_LOCK = threading.Lock()
_ACCOMMODATED = False


def _reasoning_llm_class() -> type:
    """MemOS's OpenAI LLM, with a request body a reasoning model accepts.

    MemOS sends ``temperature``, ``top_p`` and ``max_tokens`` on every
    call and builds its ``LLMConfig`` with no way to omit them, so
    gpt-5-mini answers each one with a 400. The subclass edits the body
    MemOS built and changes nothing else — same client, same parsing,
    same fallback — and leaves a non-reasoning model's body alone.
    """
    from memos.llms.openai import OpenAILLM

    class ReasoningOpenAILLM(OpenAILLM):
        """OpenAI chat completions, reasoning-model-safe."""

        def _build_request_body(self, messages: list[dict], **kwargs: Any) -> dict:
            """Drop the sampling parameters a reasoning model rejects."""
            body = super()._build_request_body(messages, **kwargs)
            if not str(body.get("model", "")).startswith(_REASONING_MODELS):
                return body
            body["max_completion_tokens"] = body.pop("max_tokens", None)
            body.pop("temperature", None)
            body.pop("top_p", None)
            return body

    return ReasoningOpenAILLM


def _propagate_context() -> None:
    """Make MemOS's thread pools carry the submitting thread's context.

    MemOS runs extraction, recall and embedding inside
    ``ContextThreadPoolExecutor``, which propagates only its own
    ``RequestContext``. The spend tracker's counters live in a
    ``ContextVar``, which those threads therefore never see: every token
    spent in one would be booked into a fresh context and dropped, and
    the run would publish a zero cost it never earned. Copying the
    context at submit time books them into the sample's own counters,
    which the tracker holds by reference.
    """
    from memos.context.context import ContextThreadPoolExecutor

    submit = ContextThreadPoolExecutor.submit

    def submit_in_context(self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
        context = contextvars.copy_context()

        def run(*inner_args: Any, **inner_kwargs: Any) -> Any:
            return context.run(fn, *inner_args, **inner_kwargs)

        return submit(self, run, *args, **kwargs)

    ContextThreadPoolExecutor.submit = submit_in_context


def _quiet_logging() -> None:
    """Move MemOS's own logging off stdout, and off INFO on disk.

    MemOS configures the root logger itself: a console handler on
    **stdout**, which `amb run`'s summary shares, and a rotating file
    handler at INFO — the level its graph store logs every Cypher
    statement and query embedding at, which is hundreds of megabytes
    over a full run. Re-running its own `configure_logging` rebuilds
    both handlers from the config dict, so patching the dict is enough;
    it is done at setup, before any MemOS thread exists to race it.

    Importing MemOS already emits a record through that stdout handler,
    before there is a config to patch, so stdlib logging is switched off
    across the import. amb's own logging is loguru, which
    `logging.disable` does not reach.
    """
    logging.disable(logging.CRITICAL)
    try:
        from memos import log as memos_log
    finally:
        logging.disable(logging.NOTSET)

    memos_log.LOGGING_CONFIG["handlers"]["console"]["stream"] = sys.stderr
    memos_log.LOGGING_CONFIG["handlers"]["file"]["level"] = "WARNING"
    memos_log.configure_logging(force=True)


def _quiet_graph_dumps() -> None:
    """Send the Neo4j store's `print()` calls to the debug log.

    Two of its methods print the Cypher they are about to run together
    with its parameters — the query embedding among them, ~30 KB per
    search and per dedup lookup — and `amb run` writes the run summary
    to stdout. `print` resolves as a module global before it resolves as
    a builtin, so binding one in that module reaches exactly those calls
    and nothing else: no stream is replaced, no other library's output
    changes, and nothing is discarded — `--log-level debug` prints all
    of it. A release that drops the prints simply stops calling this.
    """
    from memos.graph_dbs import neo4j as memos_neo4j

    def print_to_log(*args: Any, **_: Any) -> None:
        logger.bind(scope="memos").debug(" ".join(str(arg) for arg in args))

    memos_neo4j.print = print_to_log


def _accommodate(reasoning: bool) -> None:
    """Install this adapter's accommodations, once per process."""
    global _ACCOMMODATED
    with _SETUP_LOCK:
        if _ACCOMMODATED:
            return
        _quiet_logging()
        _quiet_graph_dumps()
        _propagate_context()
        if reasoning:
            from memos.llms.factory import LLMFactory

            LLMFactory.backend_to_class["openai"] = _reasoning_llm_class()
        _ACCOMMODATED = True


class MemOSMemory(Memory):
    """MemOS: a MemCube of extracted memories, held as a graph."""

    name: ClassVar[str] = "memos"
    description: ClassVar[str] = "MemOS — MemCube memory, scheduled OS-style"
    sdk_dist: ClassVar[str | None] = "MemoryOS"

    def __init__(
        self,
        model: str = DEFAULT_INGESTION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int | str = DEFAULT_EMBEDDING_DIMENSIONS,
        search_mode: str = DEFAULT_SEARCH_MODE,
        max_tokens: int | str = DEFAULT_MAX_TOKENS,
        reorganize: bool = False,
        reasoning: bool | None = None,
        **params: object,
    ) -> None:
        """Pin the models, the retrieval mode and the graph MemOS will use."""
        super().__init__(**params)
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings
        self.embedding_dimensions = int(embedding_dimensions)
        self.search_mode = search_mode
        self.max_tokens = int(max_tokens)
        # MemOS's own default: the background graph reorganization is off
        self.reorganize = reorganize
        # None auto-detects from the model name; false restores MemOS's
        # own request body for a model that accepts it
        self.reasoning = (
            self.model.startswith(_REASONING_MODELS) if reasoning is None else reasoning
        )
        # the conversation this instance was built for: MemOS is created
        # lazily, because the base contract passes the id only per call
        self._conversation_id: str | None = None
        self._mos: Any = None
        self._sessions: set[str] = set()

    def setup(self) -> None:
        """Install the accommodations MemOS needs in this harness."""
        _accommodate(self.reasoning)

    @staticmethod
    def _slug(value: str) -> str:
        """An id MemOS's own tenant-tag derivation cannot collapse."""
        return _ID_SAFE.sub("", value.lower()) or "unnamed"

    def _user(self, conversation_id: str) -> str:
        """The MemOS user one conversation's memories belong to.

        MemOS keys a search by user, and a LoCoMo conversation has two
        speakers — but the unit of memory here is the conversation, so
        both speak as one user and the speaker survives in the text of
        every message.
        """
        return f"conv{self._slug(conversation_id)}"

    def _cube(self, conversation_id: str) -> str:
        """The MemCube this conversation's memories live in, and only this one's."""
        return f"cube{self._slug(conversation_id)}"

    def _config_kwargs(self) -> dict:
        """What this benchmark overrides in MemOS's own default config."""
        return {
            "text_mem_type": "tree_text",
            "model_name": self.model,
            "max_tokens": self.max_tokens,
            "embedder_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimensions,
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
            "neo4j_password": os.environ.get("NEO4J_PASSWORD", "password"),
            # Neo4j Community has exactly one database and rejects
            # `CREATE DATABASE`, so tenancy is node-level: every node
            # carries a `user_name` and every query filters on it.
            "use_multi_db": False,
            "neo4j_auto_create": False,
            "enable_reorganize": self.reorganize,
        }

    def _store(self, conversation_id: str) -> Any:
        """The MOS instance for this conversation, built on first use."""
        if self._mos is not None:
            return self._mos
        from memos.mem_cube.general import GeneralMemCube
        from memos.mem_os.main import MOS
        from memos.mem_os.utils.default_config import (
            get_default_config,
            get_default_cube_config,
        )

        user = self._user(conversation_id)
        cube_id = self._cube(conversation_id)
        kwargs = {
            **self._config_kwargs(),
            "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            "openai_api_base": os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            "user_id": user,
            "cube_id": cube_id,
        }
        logger.bind(scope="memos").debug(
            "config: model={} embedder={} dims={} mode={} cube={}",
            self.model,
            self.embedding_model,
            self.embedding_dimensions,
            self.search_mode,
            cube_id,
        )
        with _SETUP_LOCK:
            cube = GeneralMemCube(get_default_cube_config(**kwargs))
            # MOS creates the user it is configured for, then validates it
            mos = MOS(get_default_config(**kwargs))
            mos.register_mem_cube(cube, mem_cube_id=cube_id, user_id=user)
        self._conversation_id = conversation_id
        self._mos = mos
        return mos

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Hand the session's turns to MemOS, which extracts from them."""
        if not session.turns:
            return
        mos = self._store(conversation_id)
        speakers = {turn.speaker for turn in session.turns}
        primary = sorted(speakers)[0] if speakers else None
        messages = [
            {
                "role": "user" if turn.speaker == primary else "assistant",
                "content": f"{turn.speaker}: {turn.text}",
                # MemOS puts this in the extraction prompt, and stamps
                # every message of a session with the first one it finds;
                # without it the reader stamps them `datetime.now()`
                "chat_time": session.timestamp,
                "message_id": turn.turn_id,
            }
            for turn in session.turns
        ]
        mos.add(
            messages=messages,
            user_id=self._user(conversation_id),
            mem_cube_id=self._cube(conversation_id),
            session_id=session.session_id,
        )
        self._sessions.add(session.session_id)

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
    ) -> None:
        """Hand MemOS one statement to remember (agentic mode).

        MemOS's own reader extracts from it, exactly as it does for a
        session's messages — the agent is the source, not the extractor.
        """
        mos = self._store(conversation_id)
        mos.add(
            memory_content=content,
            user_id=self._user(conversation_id),
            mem_cube_id=self._cube(conversation_id),
            session_id=session_id,
        )
        self._sessions.add(session_id)

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k memories from this conversation's MemCube."""
        if not query.strip():
            return []
        mos = self._store(conversation_id)
        cube_id = self._cube(conversation_id)
        try:
            result = mos.search(
                query,
                user_id=self._user(conversation_id),
                install_cube_ids=[cube_id],
                top_k=k,
                mode=self.search_mode,
            )
        except Exception:
            # the payload is worth having in the benchmark log, which
            # outlives the container
            logger.bind(scope="memos").error(
                "search failed: cube={!r} mode={!r} query={!r}",
                cube_id,
                self.search_mode,
                query,
            )
            raise
        hits = []
        for bucket in result.get("text_mem", []):
            for item in bucket.get("memories", []):
                metadata = item.metadata
                session_id = getattr(metadata, "session_id", None)
                hits.append(
                    MemoryHit(
                        content=item.memory,
                        # the recall stage's similarity, kept by the reranker
                        score=getattr(metadata, "relativity", None),
                        # session-level only: a memory is an extraction
                        # over a window of a session, not a verbatim turn
                        session_ids=[session_id] if session_id else [],
                        metadata={
                            "memory_id": item.id,
                            "memory_type": getattr(metadata, "memory_type", None),
                            "key": getattr(metadata, "key", None),
                        },
                    )
                )
        return hits[:k]

    def _text_memory(self) -> Any:
        """This conversation's textual memory, or None if nothing was built."""
        if self._mos is None or self._conversation_id is None:
            return None
        cube = self._mos.mem_cubes.get(self._cube(self._conversation_id))
        return getattr(cube, "text_mem", None)

    def teardown(self) -> None:
        """Delete this conversation's memories, and only this one's.

        `delete_all` is scoped to the cube's own tenant tag, so the
        conversations running beside this one under `--workers N` — nodes
        in the same shared database — are left alone. The Neo4j driver is
        closed with it: MemOS opens one per cube and exposes no close, so
        a run of many conversations would leak a connection pool each.
        """
        text_mem = self._text_memory()
        if text_mem is not None:
            text_mem.delete_all()
            driver = getattr(getattr(text_mem, "graph_store", None), "driver", None)
            if driver is not None:
                driver.close()
        self._sessions.clear()

    def stats(self) -> dict:
        """Report what this run stored, and how it retrieved."""
        stats: dict[str, Any] = {
            "sessions": len(self._sessions),
            "search_mode": self.search_mode,
            "reorganize": self.reorganize,
            "reasoning": self.reasoning,
        }
        text_mem = self._text_memory()
        if text_mem is not None:
            # per-bucket node counts: WorkingMemory / LongTermMemory / UserMemory
            stats["memories"] = text_mem.get_current_memory_size()
        return stats
