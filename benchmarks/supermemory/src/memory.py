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

"""Supermemory (supermemoryai/supermemory) — extraction memory behind an API.

Workspace package ``supermemory-benchmark`` (this directory). The member
cannot be called ``supermemory``: that is the SDK's own distribution name,
and a workspace member of the same name would shadow the dependency.

Supermemory is a *service*. The adapter is its official Python SDK
pointed at a base URL: the hosted API by default, or a Supermemory local
server (``supermemory-server``, port 6767) by setting
``SUPERMEMORY_BASE_URL`` — the two speak the same API, which is the
project's own stated design. Either way ``SUPERMEMORY_API_KEY`` is
required; the local server prints one on first boot.

Ingestion is Supermemory's own. A session is added as one *document* and
the service extracts the memories from it — there is no extraction step
in this adapter, and no ingestion model to pin, because the model doing
the extracting is the service's and lives behind its API.

Two properties of that service shape this adapter and are not obvious:

* **Writes are asynchronous.** ``documents.add`` returns as soon as the
  document is queued, and the memories it yields are not searchable until
  it reaches ``done``. A benchmark that ingested and immediately queried
  would measure an empty store, so every write is followed to a terminal
  status before the next one starts — which is also what makes the
  measured ingestion time the real one.
* **Its spend is invisible.** The extraction runs inside Supermemory's
  own infrastructure, calling a provider we never see, so no tracker in
  this harness can observe it — not the SDK-patching one (nothing goes
  through our openai client) and not the counting proxy (the traffic
  never traverses it). A supermemory run therefore reports
  ``memory_tokens_total`` 0, and that zero means *unmeasured*, not free.
  See the note in benchmarks/supermemory/README.md before quoting cost.

Each conversation is isolated by a ``conv-<id>`` *container tag*, which is
Supermemory's own scoping primitive and a hard filter on both the write
and the search — so recall cannot cross conversations by construction
rather than by post-filtering. Provenance is session-level: a returned
memory is an extraction over a document, not a verbatim turn, so the
document id maps back to the session and no turn is claimed.
"""

import os
import time
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

# the SDK's own default; SUPERMEMORY_BASE_URL points it at a local server
DEFAULT_BASE_URL = "https://api.supermemory.ai"
# statuses a document stops moving from; anything else is still in flight
TERMINAL_STATUSES = frozenset({"done", "failed"})
# how long one document may take to become searchable, and how often the
# adapter asks. A LoCoMo session is a few thousand tokens of extraction;
# the ceiling is generous so a slow queue is a slow run, not a wrong one.
INDEX_TIMEOUT_S = 600.0
INDEX_POLL_S = 1.0
# what a search asks for: "memories" is Supermemory's extracted-memory
# retrieval, the analogue of every other system's search. "documents"
# returns whole source documents and "hybrid" mixes the two.
DEFAULT_SEARCH_MODE = "memories"


class SupermemoryMemory(Memory):
    """Supermemory: server-side extraction, retrieved by container tag."""

    name: ClassVar[str] = "supermemory"
    description: ClassVar[str] = "Supermemory — extraction memory behind an API"
    sdk_dist: ClassVar[str | None] = "supermemory"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        search_mode: str = DEFAULT_SEARCH_MODE,
        rerank: bool = True,
        index_timeout: float | str = INDEX_TIMEOUT_S,
        **params: object,
    ) -> None:
        """Point the adapter at a Supermemory server and pin how it searches."""
        super().__init__(**params)
        self.base_url = base_url or os.environ.get(
            "SUPERMEMORY_BASE_URL", DEFAULT_BASE_URL
        )
        self._api_key = api_key or os.environ.get("SUPERMEMORY_API_KEY")
        self.search_mode = search_mode
        # Supermemory reranks by default; --param rerank=false measures the
        # raw retrieval underneath it
        self.rerank = rerank
        # --param values arrive as strings
        self.index_timeout = float(index_timeout)
        # the extraction model is the service's own and is not ours to pin;
        # `models()` reads these and reports None for both
        self.model: str | None = None
        self.embedding_model: str | None = None
        # document id -> session id: a hit names the documents it was
        # extracted from, and that is what restores provenance
        self._documents: dict[str, str] = {}
        # the conversation this instance was built for; teardown needs it
        # and the base contract does not pass it
        self._conversation_id: str | None = None
        self._indexed = 0

    def models(self) -> dict[str, str | None]:
        """Both are the service's own, behind its API, so neither is ours."""
        return {"ingestion_model": None, "embedding_model": None}

    @staticmethod
    def _tag(conversation_id: str) -> str:
        """The container tag every document and search of this conversation uses."""
        return f"conv-{conversation_id}"

    def setup(self) -> None:
        """Build the SDK client for the configured server.

        The key is required by both deployments — the hosted API issues
        one, and a Supermemory local server prints one on first boot — so
        a missing key fails here with that sentence rather than as a 401
        in the middle of a scored run.

        Raises:
            RuntimeError: if SUPERMEMORY_API_KEY is not set.
        """
        from supermemory import Supermemory

        if not self._api_key:
            raise RuntimeError(
                "SUPERMEMORY_API_KEY is not set. The hosted API issues one; a "
                "Supermemory local server prints one on first boot (point the "
                "adapter at it with SUPERMEMORY_BASE_URL)."
            )
        self.client = Supermemory(api_key=self._api_key, base_url=self.base_url)

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str] | None = None,
    ) -> str:
        """Add one document to the conversation, and wait for it to index.

        The container tag isolates the conversation and the metadata rides
        along as a second provenance channel beside the document id. The
        custom id carries a sequence number because agentic mode writes
        many documents per session, and a custom id is an upsert key —
        without it the session's second write would replace its first.
        """
        response = self.client.documents.add(
            content=content,
            container_tags=[self._tag(conversation_id)],
            custom_id=f"{conversation_id}:{session_id}:{len(self._documents)}",
            metadata={
                "conversation_id": conversation_id,
                "session_id": session_id,
            },
        )
        self._documents[response.id] = session_id
        self._await_indexing(response.id)
        return response.id

    def _await_indexing(self, document_id: str) -> None:
        """Block until the document is searchable, or the ceiling is hit.

        Supermemory queues a write and extracts asynchronously. Returning
        before it lands would let the first questions run against a store
        that does not hold the session yet — an ingestion failure that
        scores as a retrieval failure.

        Raises:
            RuntimeError: if the document failed, or did not reach a
                terminal status within `index_timeout`.
        """
        deadline = time.monotonic() + self.index_timeout
        while time.monotonic() < deadline:
            document = self.client.documents.get(document_id)
            if document.status in TERMINAL_STATUSES:
                if document.status == "failed":
                    raise RuntimeError(
                        f"supermemory failed to index document {document_id}"
                    )
                self._indexed += 1
                return
            time.sleep(INDEX_POLL_S)
        raise RuntimeError(
            f"supermemory did not index document {document_id} within "
            f"{self.index_timeout:.0f}s (--param index_timeout=N to raise it)"
        )

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Add the session as one document and let Supermemory extract it."""
        if not session.turns:
            return
        self._conversation_id = conversation_id
        self.store(
            conversation_id,
            "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns),
            session_id=session.session_id,
            turn_ids=[turn.turn_id for turn in session.turns],
        )

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k memories from this conversation's container.

        The question travels verbatim. `rewrite_query` is left off: it is a
        second LLM pass over the question, which is query understanding
        rather than retrieval, and it would not be comparable with the
        other systems' plain search.
        """
        if not query.strip():
            return []
        try:
            response = self.client.search.memories(
                q=query,
                container_tags=[self._tag(conversation_id)],
                limit=k,
                search_mode=self.search_mode,  # ty: ignore[invalid-argument-type]
                rerank=self.rerank,
            )
        except Exception:
            # the payload is worth having in the benchmark log, which
            # outlives the container
            logger.bind(scope="supermemory").error(
                "search failed: conversation={!r} mode={!r} query={!r}",
                conversation_id,
                self.search_mode,
                query,
            )
            raise
        hits = []
        for result in response.results:
            hits.append(
                MemoryHit(
                    content=result.memory or result.chunk or "",
                    score=result.similarity,
                    # session-level only: a memory is an extraction over a
                    # document, not a verbatim turn, so no turn is claimed
                    session_ids=self._sessions_of(result),
                    metadata={"memory_id": result.id},
                )
            )
        return hits[:k]

    def _sessions_of(self, result: Any) -> list[str]:
        """The sessions a memory was extracted from, deduped, in hit order.

        The document id is the primary channel; the metadata written with
        the document is the fallback for a result whose documents the
        service did not expand.
        """
        sessions: list[str] = []
        for document in getattr(result, "documents", None) or []:
            session_id = self._documents.get(document.id)
            if session_id is None:
                metadata = getattr(document, "metadata", None) or {}
                session_id = metadata.get("session_id")
            if session_id and session_id not in sessions:
                sessions.append(str(session_id))
        if not sessions:
            metadata = getattr(result, "metadata", None) or {}
            if session_id := metadata.get("session_id"):
                sessions.append(str(session_id))
        return sessions

    def teardown(self) -> None:
        """Delete this conversation's documents, and only this conversation's.

        Scoped to the ids this instance wrote: the containers of the
        conversations running beside it under `--workers N` are the same
        account's, and anything wider would take them with it.
        """
        for document_id in list(self._documents):
            try:
                self.client.documents.delete(document_id)
            except Exception:  # noqa: BLE001 - teardown must not fail a scored run
                logger.bind(scope="supermemory").warning(
                    "could not delete document {!r}", document_id
                )
        self._documents.clear()

    def stats(self) -> dict:
        """Report what this run stored, and how it retrieved."""
        return {
            "documents": self._indexed,
            "search_mode": self.search_mode,
            "rerank": self.rerank,
        }
