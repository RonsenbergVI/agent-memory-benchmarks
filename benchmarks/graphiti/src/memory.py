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

Workspace package ``graphiti-benchmark`` (this directory).
Needs a running Neo4j (see benchmarks/graphiti/docker-compose.yaml) and
OPENAI_API_KEY for entity extraction/embeddings. Each conversation is
isolated in its own graphiti group_id.

The models graphiti ingests with default to gpt-5-mini /
text-embedding-3-small, pinned so every system in the comparison ingests
with the same models. Override with ``--param model=...`` /
``--param embedding_model=...``; an explicit ``--param model=none`` (or
``embedding_model=none``) falls back to graphiti-core's own default.
The LLM client runs with a raised completion ceiling (``LLM_MAX_TOKENS``)
so gpt-5-mini's reasoning bursts cannot truncate structured output
mid-JSON, and teardown closes every async client before its event loop.
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

# gpt-5-mini is a reasoning model: its hidden reasoning tokens share the
# completion budget (max_output_tokens) with the visible JSON, and
# reasoning length is not deterministic call to call. graphiti-core's
# 16384 default has been exhausted in practice on an entity-summary batch
# — the JSON came back cut off mid-string (~300 chars), all retries
# truncated the same way, and the whole run died. 4x that is headroom,
# not a target: unused budget costs nothing. Same lesson as fraise's
# EXTRACTION_MAX_COMPLETION_TOKENS.
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
        # where the database lives is infrastructure, so it comes from the
        # environment (compose sets it; localhost defaults for local runs)
        self.neo4j_uri = neo4j_uri or os.environ.get(
            "NEO4J_URI", "bolt://localhost:7687"
        )
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get(
            "NEO4J_PASSWORD", "password"
        )
        # explicit none (--param model=none) leaves graphiti-core's own
        # default in place
        self.model = model
        self.embedding_model = embedding_model
        # episode uuid -> session_id: graphiti edges cite the episodes they
        # were extracted from, which is our provenance channel
        self._episodes: dict[str, str] = {}

    def setup(self) -> None:
        """Connect to Neo4j and build the graph indices."""
        from graphiti_core import Graphiti

        # one loop for the whole lifecycle: the async neo4j driver binds its
        # connection pool to the loop it first runs on
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

        Each override is omitted when its model is explicitly unset
        (``--param model=none``), so graphiti-core keeps its own default.
        """
        clients = {}
        if self.model:
            from graphiti_core.llm_client import LLMConfig, OpenAIClient

            # max_tokens goes on the client, not the LLMConfig — graphiti's
            # base client reads only its own constructor parameter
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
                episode_body=str(session),
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

        Edges extracted from the episode cite it, and the local episode map
        resolves it to its session — same provenance channel as direct
        ingestion, one episode per memory instead of per session.
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
        edges = self._await(
            self.client.search(query, group_ids=[conversation_id], num_results=k)
        )
        hits = []
        for edge in edges:
            sessions = [
                self._episodes[str(e)]
                for e in getattr(edge, "episodes", []) or []
                if str(e) in self._episodes
            ]
            hits.append(
                MemoryHit(
                    content=getattr(edge, "fact", str(edge)),
                    session_ids=sessions,
                    metadata={"uuid": str(getattr(edge, "uuid", ""))},
                )
            )
        return hits

    def teardown(self) -> None:
        """Close the graphiti client, its openai clients, and the event loop.

        The AsyncOpenAI instances inside graphiti's llm/embedder/reranker
        hold httpx pools bound to this loop; left open, their finalizers
        fire after the loop is closed and every sample ends in a spray of
        "RuntimeError: Event loop is closed". Closing them (and draining
        async generators) before the loop is the whole fix.
        """
        self._episodes.clear()
        try:
            self._await(self.client.close())
            for owner in ("llm_client", "embedder", "cross_encoder"):
                inner = getattr(getattr(self.client, owner, None), "client", None)
                close = getattr(inner, "close", None)
                if close is not None:
                    try:
                        self._await(close())
                    except Exception:  # noqa: BLE001 - best-effort cleanup
                        pass
            self._await(self._loop.shutdown_asyncgens())
        finally:
            self._loop.close()
