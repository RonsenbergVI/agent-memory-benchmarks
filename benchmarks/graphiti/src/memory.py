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
Each conversation is isolated in its own graphiti group_id. Extraction
runs on ``--llm`` (openai or anthropic, graphiti-core's own clients) and
embeddings on ``--embedding`` (openai), defaulting to the comparison's
shared pair.
"""

import asyncio
import os
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, ClassVar

from amb.base import Memory, split_model
from amb.constants import DEFAULT_EMBEDDING, DEFAULT_LLM
from amb.contracts import MemoryHit, Session

# gpt-5-mini's hidden reasoning tokens share max_output_tokens with the
# visible JSON: graphiti-core's 16384 default was exhausted on an
# entity-summary batch — JSON cut off mid-string (~300 chars), every retry
# truncated the same way, run died. 4x is headroom (unused budget costs
# nothing); same lesson as fraise's EXTRACTION_MAX_COMPLETION_TOKENS.
LLM_MAX_TOKENS = 65536
# the anthropic SDK refuses a non-streaming request this large could take >10 min
ANTHROPIC_MAX_TOKENS = 16000


class _AnthropicMessages:
    """The one Anthropic call graphiti-core makes, minus `temperature`.

    graphiti-core 0.29 sends it on every `messages.create`; anthropic 1.x
    dropped the parameter (a TypeError before any request) and Haiku 5.5
    rejects any but the default. Re-check on a graphiti-core bump.
    """

    def __init__(self) -> None:
        """Hold the real client; the key comes from ANTHROPIC_API_KEY."""
        import anthropic

        self._client = anthropic.AsyncAnthropic()
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        """Forward the request without the sampling parameter."""
        kwargs.pop("temperature", None)
        return await self._client.messages.create(**kwargs)


class GraphitiMemory(Memory):
    """Graphiti: a temporal knowledge graph over Neo4j."""

    name: ClassVar[str] = "graphiti"
    description: ClassVar[str] = "Graphiti — temporal knowledge graph (Neo4j)"
    sdk_dist: ClassVar[str | None] = "graphiti-core"
    llm_providers: ClassVar[tuple[str, ...]] = ("openai", "anthropic")
    embedding_providers: ClassVar[tuple[str, ...]] = ("openai",)

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        model: str = DEFAULT_LLM,
        embedding_model: str = DEFAULT_EMBEDDING,
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
        self.model_name(model, "llm")
        self.model_name(embedding_model, "embedding")
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
        """Return the llm_client/embedder for Graphiti(), by provider."""
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client import LLMConfig

        provider, name = split_model(self.model)
        config = LLMConfig(model=name, small_model=name)
        if provider == "anthropic":
            from graphiti_core.llm_client.anthropic_client import AnthropicClient

            # given a config, AnthropicClient reads max_tokens from it
            config.max_tokens = ANTHROPIC_MAX_TOKENS
            llm_client = AnthropicClient(config=config, client=_AnthropicMessages())
        else:
            from graphiti_core.llm_client import OpenAIClient

            # max_tokens goes on the client; LLMConfig's is ignored
            llm_client = OpenAIClient(config=config, max_tokens=LLM_MAX_TOKENS)
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                embedding_model=split_model(self.embedding_model)[1]
            )
        )
        return {"llm_client": llm_client, "embedder": embedder}

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
