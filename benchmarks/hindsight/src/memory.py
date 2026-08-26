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

"""Hindsight (vectorize-io/hindsight) — agent memory that learns.

Needs a running Hindsight server (docker-compose.yaml here), named by
HINDSIGHT_BASE_URL. Ingestion is Hindsight's own: ``retain`` hands the
server a session's transcript and its LLM extracts, so the model is the
*server's* configuration (set by compose) and is reported from the
server rather than pinned here.

Each conversation is a Hindsight *bank* — required on both ``retain``
and ``recall``, so recall cannot cross conversations by construction.
Teardown deletes the one bank, safe beside the conversations running
under ``--workers N``.

Hindsight's recall is budgeted in tokens, not hits: ``max_tokens`` is
what its API takes. The adapter asks for a budget generous enough that
k results are never the binding constraint, then takes the first k the
server ranked — comparable with the other systems, though not the
question the API is shaped around; ``--param max_tokens=N`` drives the
budget directly.
"""

import os
from typing import Any, ClassVar

from amb.base import Memory
from amb.constants import TOKEN_TRACKING_KEYS
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_BASE_URL = "http://localhost:8888"
# the budget must never limit the result count, or "k" would silently mean
# "as many as fit"; ~4k tokens covers a LoCoMo session's worth of facts
DEFAULT_MAX_TOKENS = 32000
# the server's own default ranking effort; "high" spends more on recall
DEFAULT_BUDGET = "mid"
# provenance for `observation` results (see `_sessions_of`); big enough
# that provenance is never the thing that gets truncated
DEFAULT_MAX_SOURCE_FACTS_TOKENS = 16000


class HindsightMemory(Memory):
    """Hindsight: server-side extraction, retrieved per bank."""

    name: ClassVar[str] = "hindsight"
    description: ClassVar[str] = "Hindsight — agent memory that learns"
    sdk_dist: ClassVar[str | None] = "hindsight-client"
    # `retain` reports the extraction spend, the expensive half; `recall`
    # reports nothing — its query embedding and rerank stay inside the server
    usage_coverage: ClassVar[str] = "partial"

    def __init__(
        self,
        base_url: str | None = None,
        max_tokens: int | str = DEFAULT_MAX_TOKENS,
        budget: str = DEFAULT_BUDGET,
        max_source_facts_tokens: int | str = DEFAULT_MAX_SOURCE_FACTS_TOKENS,
        timeout: float | str = 300.0,
        **params: object,
    ) -> None:
        """Point the adapter at a Hindsight server and pin how it recalls."""
        super().__init__(**params)
        self.base_url = os.environ.get("HINDSIGHT_BASE_URL", DEFAULT_BASE_URL)
        if base_url:
            self.base_url = base_url
        # a secret; environment only, never --param
        self._api_key = os.environ.get("HINDSIGHT_API_KEY")
        # --param values arrive as strings
        self.max_tokens = int(max_tokens)
        self.budget = budget
        self.max_source_facts_tokens = int(max_source_facts_tokens)
        self.timeout = float(timeout)
        # Read here, not in setup(): the Runner's probe instance asks for
        # models()/version() without calling setup() — populating there once
        # recorded no ingestion model at all in a run's identity. Compose
        # sets these alongside the ones it gives the server.
        self.model = os.environ.get("HINDSIGHT_API_LLM_MODEL")
        self.embedding_model = os.environ.get("HINDSIGHT_API_EMBEDDING_MODEL")
        # document id -> session id. `retain` takes the document id, so
        # this is chosen rather than discovered, and a hit names it back.
        self._documents: dict[str, str] = {}
        self._conversation_id: str | None = None
        self._retained = 0
        # server-reported spend, read by TiktokenUsageTracker
        self._usage: dict[str, int] = dict.fromkeys(TOKEN_TRACKING_KEYS, 0)

    def usage_counters(self) -> dict:
        """The spend hindsight reported for this instance so far (copy)."""
        return dict(self._usage)

    def _count_reported_usage(self, response: object) -> None:
        """Book what the server said one retain cost.

        Billed from the usage hindsight reports on the response, not
        estimated. `thoughts_tokens` are already inside `output_tokens` —
        deliberately not added again.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage["llm_calls"] += 1
        self._usage["llm_input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self._usage["llm_output_tokens"] += getattr(usage, "output_tokens", 0) or 0

    def models(self) -> dict[str, str | None]:
        """The models Hindsight extracts and embeds with, as the server has them."""
        return {"ingestion_model": self.model, "embedding_model": self.embedding_model}

    def version(self) -> str | None:
        """The server's version — the server is the system under test.

        Builds its own client when there is none: the Runner probes this
        before `setup()`, and the client-SDK fallback would record the
        client line, which drifts (0.9.1 client against a 0.9.2 server)
        into `system_version`, part of the run's identity.
        """
        client = getattr(self, "client", None)
        try:
            if client is None:
                from hindsight_client import Hindsight

                with Hindsight(
                    base_url=self.base_url, api_key=self._api_key, timeout=self.timeout
                ) as probe:
                    version = probe.get_version()
            else:
                version = client.get_version()
        except Exception:  # noqa: BLE001 - fall back to the client SDK's dist
            return super().version()
        for attr in ("version", "api_version", "server_version"):
            if value := getattr(version, attr, None):
                return str(value)
        return super().version()

    @staticmethod
    def _bank_id(conversation_id: str) -> str:
        """The bank this conversation's memories live in, and only this one's."""
        return f"conv-{conversation_id}"

    def setup(self) -> None:
        """Connect to the Hindsight server."""
        from hindsight_client import Hindsight

        self.client = Hindsight(
            base_url=self.base_url, api_key=self._api_key, timeout=self.timeout
        )

    def _ensure_bank(self, conversation_id: str) -> str:
        """Create this conversation's bank if the server does not have it yet."""
        bank_id = self._bank_id(conversation_id)
        if self._conversation_id == conversation_id:
            return bank_id
        try:
            self.client.create_bank(bank_id=bank_id)
        except Exception:  # noqa: BLE001 - an existing bank is the good case
            logger.bind(scope="hindsight").debug(
                "bank {!r} already exists or could not be created", bank_id
            )
        self._conversation_id = conversation_id
        return bank_id

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str] | None = None,
    ) -> str:
        """Retain one document in the conversation's bank.

        The document id is chosen rather than left to the server, so a hit
        names its session back without a lookup; the metadata carries the
        same thing as a second channel.
        """
        bank_id = self._ensure_bank(conversation_id)
        document_id = f"{conversation_id}:{session_id}:{len(self._documents)}"
        metadata: dict[str, object] = {
            "conversation_id": conversation_id,
            "session_id": session_id,
        }
        if turn_ids is not None:
            metadata["turn_ids"] = turn_ids
        response = self.client.retain(
            bank_id=bank_id,
            content=content,
            document_id=document_id,
            metadata=metadata,
        )
        self._count_reported_usage(response)
        self._documents[document_id] = session_id
        self._retained += 1
        return document_id

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Retain the session's transcript; the server extracts from it."""
        if not session.turns:
            return
        self.store(
            conversation_id,
            "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns),
            session_id=session.session_id,
            turn_ids=[turn.turn_id for turn in session.turns],
        )

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k recalled memories from this conversation's bank.

        The question travels verbatim. See the module docstring on why the
        token budget is set high and k is applied here instead.
        """
        if not query.strip():
            return []
        bank_id = self._ensure_bank(conversation_id)
        try:
            response = self.client.recall(
                bank_id=bank_id,
                query=query,
                max_tokens=self.max_tokens,
                budget=self.budget,
                # not a retrieval knob — the provenance channel for results
                # with no document of their own (see `_sessions_of`)
                include_source_facts=True,
                max_source_facts_tokens=self.max_source_facts_tokens,
            )
        except Exception:
            # log the payload — the benchmark log outlives the container
            logger.bind(scope="hindsight").error(
                "recall failed: bank={!r} query={!r}",
                self._bank_id(conversation_id),
                query,
            )
            raise
        hits = []
        for result in (response.results or [])[:k]:
            hits.append(
                MemoryHit(
                    content=result.text,
                    score=self._score(result),
                    # session-level only: a result is an extracted memory
                    # over a document, not a verbatim turn
                    session_ids=self._sessions_of(result, response.source_facts or {}),
                    metadata={"result_id": result.id, "type": result.type},
                )
            )
        return hits

    @staticmethod
    def _score(result: Any) -> float | None:
        """The result's ranking score, whichever of them the server filled."""
        scores = getattr(result, "scores", None)
        if scores is None:
            return None
        for attr in ("final", "combined", "score", "relevance", "semantic"):
            value = getattr(scores, attr, None)
            if isinstance(value, int | float):
                return float(value)
        return None

    def _sessions_of(self, result: Any, source_facts: dict) -> list[str]:
        """The sessions a recalled memory came from, deduped, in hit order.

        A ``world`` memory names the document it was extracted from, so
        `document_id` (or the metadata beside it) answers directly. An
        ``observation`` is synthesised across sources — `document_id`,
        `metadata` and `chunk_id` all None — and is attributable only
        through its source facts, which is what `include_source_facts`
        buys. Without that hop roughly half of a run's hits carry no
        provenance, and the harness drops those questions rather than
        scoring them 0.0 — silently reporting the easier half as the
        whole run.
        """
        if direct := self._session_of_one(result):
            return [direct]
        sessions: list[str] = []
        for fact_id in getattr(result, "source_fact_ids", None) or []:
            fact = source_facts.get(str(fact_id))
            if fact is None:
                continue
            if (
                session_id := self._session_of_one(fact)
            ) and session_id not in sessions:
                sessions.append(session_id)
        return sessions

    def _session_of_one(self, item: Any) -> str | None:
        """One document-backed item's session: by document id, then metadata."""
        document_id = getattr(item, "document_id", None)
        if document_id and (session_id := self._documents.get(str(document_id))):
            return session_id
        metadata = getattr(item, "metadata", None) or {}
        if session_id := metadata.get("session_id"):
            return str(session_id)
        return None

    def teardown(self) -> None:
        """Delete this conversation's bank, and only this conversation's.

        The conversations running beside this one under `--workers N`
        share the server; anything wider would take them with it.
        """
        if self._conversation_id is not None:
            try:
                self.client.delete_bank(bank_id=self._bank_id(self._conversation_id))
            except Exception:  # noqa: BLE001 - teardown must not fail a scored run
                logger.bind(scope="hindsight").warning(
                    "could not delete bank {!r}", self._bank_id(self._conversation_id)
                )
        self._documents.clear()
        if client := getattr(self, "client", None):
            client.close()

    def stats(self) -> dict:
        """Report what this run stored, and how it recalled."""
        return {
            "documents": self._retained,
            "max_tokens": self.max_tokens,
            "budget": self.budget,
        }
