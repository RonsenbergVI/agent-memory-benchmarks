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

"""Graphiti (getzep/graphiti) — temporal knowledge graph memory on Neo4j.

Needs a running Neo4j (docker-compose.yaml here) and OPENAI_API_KEY.
Each conversation is isolated in its own graphiti group_id. Ingestion
models are pinned to gpt-5-mini / text-embedding-3-small so every system
in the comparison ingests with the same models; override with
``--param model=...`` / ``--param embedding_model=...``, or ``none`` to
fall back to graphiti-core's own default.
"""

import asyncio
import os
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# gpt-5-mini's hidden reasoning tokens share max_output_tokens with the
# visible JSON: graphiti-core's 16384 default was exhausted on an
# entity-summary batch — JSON cut off mid-string (~300 chars), every retry
# truncated the same way, run died. 4x is headroom (unused budget costs
# nothing); same lesson as fraise's EXTRACTION_MAX_COMPLETION_TOKENS.
LLM_MAX_TOKENS = 65536


class GraphitiMemory(Memory):
    """Graphiti: a temporal knowledge graph over Neo4j."""

    name: ClassVar[str] = "graphiti"
    description: ClassVar[str] = "Graphiti — temporal knowledge graph (Neo4j)"
    sdk_dist: ClassVar[str | None] = "graphiti-core"

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        model: str | None = DEFAULT_INGESTION_MODEL,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        **params: object,
    ) -> None:
        """Pin the Neo4j connection and the models graphiti ingests with."""
        super().__init__(**params)
        # infrastructure comes from the environment (compose sets it)
        self.neo4j_uri = neo4j_uri or os.environ.get(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get(
            "NEO4J_PASSWORD", "password"
        )
        self.model = model
        self.embedding_model = embedding_model
        # episode uuid -> session_id: edges cite episodes = our provenance channel
        self._episodes: dict[str, str] = {}

    def setup(self) -> None:
        """Connect to Neo4j and build the graph indices."""
        from graphiti_core import Graphiti

        # one loop for the lifecycle: async neo4j binds its pool to its first loop
        self._loop = asyncio.new_event_loop()
        self.client = Graphiti(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password,
            **self._model_clients(),
        )
        self._await(self.client.build_indices_and_constraints())

    def _model_clients(self) -> dict:
        """Return llm_client/embedder overrides for Graphiti().

        An explicitly unset model (``--param model=none``) omits its
        override, so graphiti-core keeps its own default.
        """
        clients = {}
        if self.model:
            from graphiti_core.llm_client import LLMConfig, OpenAIClient

            # max_tokens goes on the client; LLMConfig's is ignored
            clients["llm_client"] = OpenAIClient(
                config=LLMConfig(model=self.model, small_model=self.model),
                max_tokens=LLM_MAX_TOKENS,
            )
        if self.embedding_model:
            from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

            clients["embedder"] = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(embedding_model=self.embedding_model)
            )
        return clients

    def _await[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return self._loop.run_until_complete(coro)

    @staticmethod
    def _reference_time(session: Session) -> datetime:
        if session.timestamp:
            for fmt in ("%I:%M %p on %d %B, %Y", "%Y/%m/%d (%a) %H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(session.timestamp, fmt).replace(tzinfo=UTC)
                except ValueError:
                    continue
        return datetime.now(UTC)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Add the session to the graph as one episode."""
        from graphiti_core.nodes import EpisodeType

        result = self._await(
            self.client.add_episode(
                name=f"{conversation_id}:{session.session_id}",
                # turns, no Session.__str__ header; time goes via reference_time
                episode_body="\n".join(
                    f"{turn.speaker}: {turn.text}" for turn in session.turns
                ),
                source=EpisodeType.message,
                source_description="benchmark conversation session",
                reference_time=self._reference_time(session),
                group_id=conversation_id,
            )
        )
        episode = getattr(result, "episode", result)
        uuid = getattr(episode, "uuid", None)
        if uuid is not None:
            self._episodes[str(uuid)] = session.session_id

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session: Session,
    ) -> None:
        """Add one agent-authored memory as its own episode (agentic mode).

        Same provenance channel as ingest_session, one episode per memory
        instead of per session.
        """
        from graphiti_core.nodes import EpisodeType

        result = self._await(
            self.client.add_episode(
                name=f"{conversation_id}:{session.session_id}",
                episode_body=content,
                source=EpisodeType.text,
                source_description="agent-authored memory",
                reference_time=self._reference_time(session),
                group_id=conversation_id,
            )
        )
        episode = getattr(result, "episode", result)
        uuid = getattr(episode, "uuid", None)
        if uuid is not None:
            self._episodes[str(uuid)] = session.session_id

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best graph edges (facts) for the query."""
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_RRF,
        )

        config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        config.limit = k
        results = self._await(
            self.client.search_(query, config=config, group_ids=[conversation_id])
        )
        hits = []
        for edge in results.edges:
            sessions = [
                self._episodes[str(e)]
                for e in getattr(edge, "episodes", []) or []
                if str(e) in self._episodes
            ]
            hits.append(
                MemoryHit(
                    content=edge.fact,
                    session_ids=sessions,
                    metadata={"uuid": str(edge.uuid)},
                )
            )
        return hits

    def teardown(self) -> None:
        """Close the graphiti client and its event loop.

        Deliberately leaves graphiti's AsyncOpenAI clients open: closing
        them per sample was the prime suspect in SIGSEGV/SIGABRT crashes
        on the high-churn longmemeval cells. Leaving them only sprays
        harmless "Event loop is closed" from GC finalizers; revisit only
        with a faulthandler-attributed stack.
        """
        self._episodes.clear()
        try:
            self._await(self.client.close())
        finally:
            self._loop.close()
