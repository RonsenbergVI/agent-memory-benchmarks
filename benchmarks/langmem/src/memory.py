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

"""LangMem (langchain-ai/langmem) — extraction over a LangGraph store.

Workspace package ``langmem-benchmark`` (this directory). LangMem runs
in-process and is deliberately store-agnostic: ``create_memory_store_manager``
reads and writes a LangGraph ``BaseStore``, and the store is what holds
the memories and searches them. The one measured here is
``PostgresStore`` — Postgres with pgvector, the store LangMem's own docs
reach for outside a notebook — named by ``LANGMEM_DATABASE_URL``.

The manager is used standalone rather than inside a LangGraph agent, so
direct mode stays a straight ingest/search measurement: each session's
messages go through ``manager.invoke``, which searches for related
memories, extracts new ones with the configured model, and applies the
inserts and updates to the store.

Two properties shape this adapter and are not obvious:

* **Provenance comes from the namespace.** A LangMem memory is an
  extraction over a window of messages, so there is nothing on the
  stored value tying it to a turn — but the manager writes under a
  templated namespace, and the store searches a namespace *prefix*.
  Writing each session under ``("memories", conversation, session)`` and
  searching ``("memories", conversation)`` therefore spans a whole
  conversation while every hit still names the session it came from.
* **Search is the store's, by LangMem's own design.**
  ``create_search_memory_tool`` is a thin wrapper over
  ``store.search(namespace, query=..., limit=...)``, so direct mode calls
  that same method with the same arguments instead of routing a tool call
  through an agent. ``create_memory_searcher`` — LangMem's other search
  surface — puts an LLM query-rewriting pass in front of it, which is
  query understanding rather than retrieval and would not be comparable
  with the other systems.

Each conversation is its own namespace subtree, so recall cannot cross
conversations and teardown deletes one conversation's keys without
touching the conversations running beside it under ``--workers N``.
"""

import os
import threading
import uuid
from contextlib import ExitStack
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_DATABASE_URL = "postgresql://langmem:langmem@localhost:5432/langmem"
# LangMem's own default: how many existing memories the manager pulls in
# as context before extracting, which is what lets it update and
# consolidate rather than only append.
DEFAULT_QUERY_LIMIT = 5
# the root segment every namespace in this benchmark hangs off, matching
# the first segment of LangMem's own default namespace
ROOT = "memories"
# the store's migrations are check-then-create (including `CREATE
# EXTENSION vector`); workers arriving together race them.
_SETUP_LOCK = threading.Lock()


class LangMemMemory(Memory):
    """LangMem: LLM extraction and consolidation over a LangGraph store."""

    name: ClassVar[str] = "langmem"
    description: ClassVar[str] = "LangMem — extraction over a LangGraph store"
    sdk_dist: ClassVar[str | None] = "langmem"

    def __init__(
        self,
        model: str = DEFAULT_INGESTION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int | str = DEFAULT_EMBEDDING_DIMENSIONS,
        query_limit: int | str = DEFAULT_QUERY_LIMIT,
        **params: object,
    ) -> None:
        """Pin the models and the consolidation context LangMem will use."""
        super().__init__(**params)
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings
        self.embedding_dimensions = int(embedding_dimensions)
        self.query_limit = int(query_limit)
        self._stack = ExitStack()
        self._store: Any = None
        self._manager: Any = None
        # the conversations this instance wrote to; teardown needs them
        # and the base contract does not pass them
        self._conversations: set[str] = set()
        self._sessions: set[str] = set()

    def setup(self) -> None:
        """Open the store, run its migrations, and build the manager."""
        from langgraph.store.postgres import PostgresStore
        from langmem import create_memory_store_manager

        url = os.environ.get("LANGMEM_DATABASE_URL", DEFAULT_DATABASE_URL)
        index = {
            "dims": self.embedding_dimensions,
            # LangGraph resolves this through langchain's `init_embeddings`,
            # so the embedder is the openai SDK and its spend is measured
            "embed": f"openai:{self.embedding_model}",
        }
        logger.bind(scope="langmem").debug(
            "store: {} | model={} embedder={} dims={}",
            url.rsplit("@", 1)[-1],  # never log the credentials
            self.model,
            self.embedding_model,
            self.embedding_dimensions,
        )
        with _SETUP_LOCK:
            self._store = self._stack.enter_context(
                PostgresStore.from_conn_string(url, index=index)
            )
            self._store.setup()
        self._manager = create_memory_store_manager(
            self.model,
            # templated per call: the session segment is what makes a hit
            # attributable, and the conversation segment is what keeps
            # recall inside one conversation
            namespace=(ROOT, "{conversation}", "{session}"),
            store=self._store,
            query_limit=self.query_limit,
        )

    @staticmethod
    def _config(conversation_id: str, session_id: str) -> dict:
        """The runtime config LangMem fills its namespace template from."""
        return {
            "configurable": {"conversation": conversation_id, "session": session_id}
        }

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Hand the session's turns to LangMem, which extracts from them."""
        if not session.turns:
            return
        speakers = {turn.speaker for turn in session.turns}
        primary = sorted(speakers)[0] if speakers else None
        messages = [
            {
                "role": "user" if turn.speaker == primary else "assistant",
                "content": f"{turn.speaker}: {turn.text}",
            }
            for turn in session.turns
        ]
        self._manager.invoke(
            {"messages": messages},
            config=self._config(conversation_id, session.session_id),
        )
        self._conversations.add(conversation_id)
        self._sessions.add(session.session_id)

    def store_memory(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
    ) -> None:
        """Write one memory verbatim (agentic mode).

        This is what LangMem's own `manage_memory` tool does — a `put`
        of ``{"content": ...}`` under the namespace, with no extraction
        pass. The agent is the extractor here, so the memory lands as
        written.
        """
        self._store.put(
            (ROOT, conversation_id, session_id),
            key=str(uuid.uuid4()),
            value={"content": content},
        )
        self._conversations.add(conversation_id)
        self._sessions.add(session_id)

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k memories from this conversation's namespace."""
        if not query.strip():
            return []
        items = self._store.search((ROOT, conversation_id), query=query, limit=k)
        hits = []
        for item in items:
            hits.append(
                MemoryHit(
                    content=self._content(item.value),
                    score=item.score,
                    # session-level only: a memory is an extraction over a
                    # window of messages, not a verbatim turn
                    session_ids=[item.namespace[-1]] if len(item.namespace) > 2 else [],
                    metadata={"key": item.key, "kind": item.value.get("kind")},
                )
            )
        return hits

    @staticmethod
    def _content(value: dict) -> str:
        """The memory's text, whether it is a string or a schema instance.

        LangMem stores unstructured memories as ``{"content": "..."}`` and
        structured ones (a `schemas=` model) as a dumped dict under the
        same key, so both shapes arrive here.
        """
        content = value.get("content", value)
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return "\n".join(f"{k}: {v}" for k, v in content.items() if v)
        return str(content)

    def teardown(self) -> None:
        """Delete this run's memories, then close the store's pool.

        Scoped to the conversations this instance wrote: the namespaces of
        the conversations running beside it under `--workers N` are
        siblings in the same table, and a wider sweep would take them.
        """
        for conversation_id in self._conversations:
            while batch := self._store.search(
                (ROOT, conversation_id), limit=100, refresh_ttl=False
            ):
                for item in batch:
                    self._store.delete(item.namespace, item.key)
        self._conversations.clear()
        self._sessions.clear()
        self._stack.close()
        self._store = None
        self._manager = None

    def stats(self) -> dict:
        """Report what this run stored, and how it consolidated."""
        stored = 0
        for conversation_id in self._conversations:
            offset = 0
            while batch := self._store.search(
                (ROOT, conversation_id), limit=100, offset=offset, refresh_ttl=False
            ):
                stored += len(batch)
                offset += len(batch)
        return {
            "sessions": len(self._sessions),
            "memories": stored,
            "query_limit": self.query_limit,
        }
