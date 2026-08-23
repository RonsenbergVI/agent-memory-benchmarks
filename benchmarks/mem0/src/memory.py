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

"""Mem0 (mem0ai/mem0) — vector store + LLM fact extraction.

Workspace package ``mem0-benchmark`` (this directory); like every
integration, its module is ``src`` since SDKs own the pretty import names.
Vectors go to the Qdrant server named by QDRANT_HOST/QDRANT_PORT (set by the
Docker environment in benchmarks/mem0/); without those it falls back to
mem0's embedded local store. Extraction/embeddings use OpenAI (needs
OPENAI_API_KEY).

The models mem0 uses to ingest come from the command line —
``--param model=... --param embedding_model=...`` — falling back to the
defaults below. Pass ``--param config=...`` (a full Memory.from_config
dict) to bypass all of it and point at other backends.
"""

import os
import threading
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# parallel workers each build a client, and every mem0 Qdrant store does
# check-then-create on the same collection — concurrent setups race that
# gap and the losers die on 409 Conflict. One build at a time; after the
# first, the collection exists for everyone.
_SETUP_LOCK = threading.Lock()


class Mem0Memory(Memory):
    """Mem0: vector store plus LLM fact extraction."""

    name: ClassVar[str] = "mem0"
    description: ClassVar[str] = "Mem0 — vector + LLM extraction memory"
    sdk_dist: ClassVar[str | None] = "mem0ai"

    def __init__(
        self,
        config: dict | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        reasoning: bool = True,
        **params: object,
    ) -> None:
        """Pin the extraction and embedding models mem0 will use."""
        super().__init__(**params)
        self._config = config
        self.model = model or DEFAULT_INGESTION_MODEL
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        # mem0's name-based reasoning detection misses gpt-5-mini, so it sends
        # temperature/top_p and the model rejects them (400). The explicit flag
        # selects the parameter set that omits them, which is safe for
        # non-reasoning models too — set False to restore mem0's sampling args.
        self.reasoning = reasoning
        self._conversations: set[str] = set()

    def setup(self) -> None:
        """Build the mem0 client, wiring it to Qdrant when one is configured."""
        from mem0 import Memory

        config: dict[str, Any] | None = self._config
        if config is None:
            config = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self.model,
                        "is_reasoning_model": self.reasoning,
                    },
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": self.embedding_model},
                },
            }
            # where the vector store lives is infrastructure, so it comes
            # from the environment (compose sets it); without it mem0 falls
            # back to its embedded local store
            if host := os.environ.get("QDRANT_HOST"):
                config["vector_store"] = {
                    "provider": "qdrant",
                    "config": {
                        "host": host,
                        "port": int(os.environ.get("QDRANT_PORT", "6333")),
                    },
                }
        logger.bind(scope="mem0").debug(
            "config: llm={} embedder={} vector_store={}",
            (config or {}).get("llm", {}).get("config", {}).get("model"),
            (config or {}).get("embedder", {}).get("config", {}).get("model"),
            (config or {})
            .get("vector_store", {})
            .get("config", {})
            .get("host", "embedded"),
        )
        with _SETUP_LOCK:
            self.memory = Memory.from_config(config) if config else Memory()

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Hand the session to mem0, which extracts facts from it."""
        speakers = {t.speaker for t in session.turns}
        primary = sorted(speakers)[0] if speakers else None
        messages = [
            {
                "role": "user" if turn.speaker == primary else "assistant",
                "content": f"{turn.speaker}: {turn.text}",
            }
            for turn in session.turns
        ]
        if not messages:
            return
        self.memory.add(
            messages,
            user_id=conversation_id,
            metadata={"session_id": session.session_id, "timestamp": session.timestamp},
        )
        self._conversations.add(conversation_id)

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str],
        timestamp: str | None = None,
    ) -> None:
        """Store one agent-authored memory verbatim (agentic mode).

        `infer=False` skips mem0's own LLM extraction: in agentic mode the
        *agent* is the extractor, so the fact lands as written, with its
        provenance in the metadata search reads back.
        """
        self.memory.add(
            content,
            user_id=conversation_id,
            infer=False,
            metadata={
                "session_id": session_id,
                "turn_ids": list(turn_ids),
                "timestamp": timestamp,
            },
        )
        self._conversations.add(conversation_id)

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return mem0's k best memories for the query."""
        # mem0 2.0 API: entity scoping goes in `filters`, result count is
        # `top_k` (`limit` would be silently swallowed by **kwargs)
        response = self.memory.search(
            query, filters={"user_id": conversation_id}, top_k=k
        )
        results = (
            response.get("results", []) if isinstance(response, dict) else response
        )
        logger.bind(scope="mem0").debug(
            "search {!r} -> {} results", query[:50], len(results)
        )
        hits = []
        for r in results:
            metadata = r.get("metadata") or {}
            session_id = metadata.get("session_id")
            # turn ids exist only on agentic-mode memories, whose provenance
            # is exact; mem0's own extraction spans a session
            turn_ids = metadata.get("turn_ids") or []
            hits.append(
                MemoryHit(
                    content=r.get("memory", ""),
                    score=r.get("score"),
                    turn_ids=[str(t) for t in turn_ids],
                    session_ids=[str(session_id)] if session_id is not None else [],
                    metadata=metadata,
                )
            )
        return hits

    def teardown(self) -> None:
        """Delete every memory this run created."""
        for conversation_id in self._conversations:
            self.memory.delete_all(user_id=conversation_id)
        self._conversations.clear()
