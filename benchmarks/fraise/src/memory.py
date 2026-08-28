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

"""Fraise (RonsenbergVI/fraise) — hybrid full-text + graph memory database.

Needs a running fraise server (docker-compose.yaml here), named by
FRAISE_BASE_URL. The SDK has no extractor, so direct-mode ingestion runs
one in the adapter: gpt-5-mini (the same ingestion model as every other
system) distills each session into facts, and the SDK's OpenAIEmbedder
(text-embedding-3-small) encodes facts and queries in-process, so token
spend is visible to the OpenAIUsageTracker. In agentic mode the driving
agent is the extractor. ``--param model=none`` restores raw per-turn
ingestion; ``--param embedding_model=none`` drops the embedder; both give
fraise's stock hybrid retrieval with no LLM or embedding calls.

A `conv-<id>` topic anchor isolates each conversation (FQL anchors are
hard filters); a local value -> (turns, session) map restores provenance,
since hits return only value/score/timestamp. The alpha SDK has no
delete, so teardown is a no-op — the server runs without a volume and
each `up` starts empty.
"""

import json
import re
from typing import TYPE_CHECKING, ClassVar, cast

from amb.base import Memory
from amb.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_INGESTION_MODEL
from amb.contracts import MemoryHit, Session
from amb.logs import logger, quiet_frameworks
from src.settings import Settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.completion_create_params import ResponseFormat

# gpt-5-mini's hidden reasoning tokens share the completion budget and vary
# call to call; uncapped, a long reasoning pass truncated the JSON mid-string
# (observed json.JSONDecodeError). Sessions measured ran ~2.7k total tokens
# (~2k reasoning), so 16k is generous headroom.
EXTRACTION_MAX_COMPLETION_TOKENS = 16000

# graph-walk hops from a match on recall
DEFAULT_RECALL_DEPTH = 2

EXTRACTION_SYSTEM_PROMPT = """\
You are an assistant with a long-term memory. You have just had the
conversation below and now store what is worth remembering. Extract every
concrete fact, event, preference, date, and plan a future question could
ask about, each as a standalone statement that a search could find on its
own; skip filler and pleasantries. Cite the turn markers each fact came
from exactly as shown in the transcript, but never write those markers into
the fact itself. Name the people, places, and things each fact mentions in
its entities."""

_MEMORIES_SCHEMA = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "turn_ids": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["fact", "turn_ids", "entities"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}


class FraiseMemory(Memory):
    """Fraise: hybrid full-text + graph retrieval over per-turn facts."""

    name: ClassVar[str] = "fraise"
    description: ClassVar[str] = "Fraise — hybrid full-text + graph memory database"
    sdk_dist: ClassVar[str | None] = "fraise-sdk"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = DEFAULT_INGESTION_MODEL,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int | str | None = None,
        depth: int | str | None = DEFAULT_RECALL_DEPTH,
        **params: object,
    ) -> None:
        """Point the adapter at a fraise server and pin its ingestion models."""
        super().__init__(**params)
        self.base_url = base_url or Settings().base_url
        # explicit none (--param model=none) restores raw per-turn ingestion
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings; the SDK wants an int or None
        self.embedding_dimensions = (
            int(embedding_dimensions) if embedding_dimensions else None
        )
        # --param depth=none restores the server's own default
        self.depth = int(depth) if depth else None
        # hits return only the value; this map restores turn/session provenance
        self._provenance: dict[tuple[str, str], tuple[list[str], str]] = {}

    def version(self) -> str | None:
        """The fraise server's version — that is the system under test."""
        from fraise_sdk import FraiseClient

        try:
            with FraiseClient(self.base_url) as client:
                server = client.server_version()
            if server:
                return str(server)
        except Exception:  # noqa: BLE001 - any failure falls back to the SDK
            pass
        return super().version()

    def setup(self) -> None:
        """Connect to the fraise server, with an embedder when one is configured."""
        from fraise_sdk import FraiseClient

        embedder = None
        if self.embedding_model:
            from fraise_sdk.providers import OpenAIEmbedder

            embedder = OpenAIEmbedder(
                model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
        self.client = FraiseClient(self.base_url, embedder=embedder)
        if self.model:
            from openai import OpenAI

            self._openai = OpenAI()
        quiet_frameworks("openai", "httpx")

    @staticmethod
    def _tokens(values: list[str] | None) -> list[str]:
        """Free-form topics/entities as FQL tokens.

        The grammar splits on whitespace, so multi-word values are
        hyphenated to survive as a single filter.
        """
        tokens = (re.sub(r"[^A-Za-z0-9]+", "-", v).strip("-") for v in values or [])
        return [t for t in tokens if t]

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str],
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> None:
        """Remember one value with the forced conversation/session anchors.

        `turn_ids` feed the local provenance map that search reads back.
        """
        # FQL phrases have no apostrophe escape; the typographic swap
        # round-trips identically, which the provenance map relies on
        value = content.replace("'", "’")
        remember_topics = [
            f"conv-{conversation_id}",
            f"session-{session_id}",
            *self._tokens(topics),
        ]
        remember_entities = self._tokens(entities)
        try:
            self.client.remember(
                value, topics=remember_topics, entities=remember_entities
            )
        except Exception:
            # a 400 here has been a live FQL-grammar edge case (e.g. "found
            # top"); log the exact payload so the offending input is diagnosable
            logger.bind(scope="fraise").error(
                "remember failed: value={!r} topics={!r} entities={!r}",
                value,
                remember_topics,
                remember_entities,
            )
            raise
        self._provenance[(conversation_id, value)] = (list(turn_ids), session_id)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Store the session: extracted facts by default, raw turns without a model.

        The extractor's cited turn markers become each fact's provenance;
        entities feed the server's graph walk.
        """
        if not session.turns:
            return
        if not self.model:
            for turn in session.turns:
                self.store(
                    conversation_id,
                    f"{turn.speaker}: {turn.text}",
                    session_id=session.session_id,
                    turn_ids=[turn.turn_id],
                    entities=[turn.speaker],
                )
            return
        known = {turn.turn_id for turn in session.turns}
        for memory in self._extract_memories(conversation_id, session):
            fact = memory["fact"].strip()
            if not fact:
                continue
            self.store(
                conversation_id,
                fact,
                session_id=session.session_id,
                turn_ids=[t for t in memory["turn_ids"] if t in known],
                entities=[e for e in memory["entities"] if e.strip()],
            )

    def _extract_memories(self, conversation_id: str, session: Session) -> list[dict]:
        """One extraction call: the session's transcript in, its facts out.

        A truncated/malformed response is logged and treated as no facts —
        matching mem0's extractor, which skips the same failure rather than
        failing the run.
        """
        timestamp = f" of {session.timestamp}" if session.timestamp else ""
        transcript = f"Conversation{timestamp}\n" + "\n".join(
            f"{turn.turn_id} | {turn.speaker}: {turn.text}" for turn in session.turns
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        # the cast bridges the nested-schema dict the checker cannot narrow
        response_format = cast(
            "ResponseFormat",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "memories",
                    "strict": True,
                    "schema": _MEMORIES_SCHEMA,
                },
            },
        )
        # setup() builds self._openai only when a model is set
        assert self.model is not None
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=EXTRACTION_MAX_COMPLETION_TOKENS,
            response_format=response_format,
        )
        choice = response.choices[0]
        # refusal/empty completion has content=None — same exit as bad JSON
        if choice.message.content is not None:
            try:
                return json.loads(choice.message.content)["memories"]
            except json.JSONDecodeError:
                pass
        logger.bind(scope="fraise").warning(
            "{}/{}: extraction response unparseable (finish_reason={}); "
            "skipping this session's facts",
            conversation_id,
            session.session_id,
            choice.finish_reason,
        )
        return []

    def recall_hits(
        self,
        conversation_id: str,
        *,
        query: str,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
        k: int = 10,
    ) -> list[MemoryHit]:
        """Recall inside the conversation, with provenance restored.

        `query` travels as one quoted phrase term, never bare keywords —
        literal inside its quotes, so it cannot collide with the grammar's
        reserved words — and doubles as the embed text. `self.depth` bounds
        the graph hops a match may walk.
        """
        recall_topics = [f"conv-{conversation_id}", *(topics or [])]
        try:
            result = self.client.recall(
                query=query,
                topics=recall_topics,
                entities=entities or None,
                top=k,
                depth=self.depth,
            )
        except Exception:
            # log the exact inputs — the benchmark log outlives the server
            logger.bind(scope="fraise").error(
                "recall failed: query={!r} topics={!r} entities={!r}",
                query,
                recall_topics,
                entities,
            )
            raise
        hits = []
        for hit in result.hits:
            turn_ids, session_id = self._provenance.get(
                (conversation_id, hit.value), ([], None)
            )
            hits.append(
                MemoryHit(
                    content=hit.value,
                    score=hit.score,
                    turn_ids=turn_ids,
                    session_ids=[session_id] if session_id else [],
                )
            )
        return hits

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best facts for the query, inside the conversation."""
        if not query.strip():
            return []
        return self.recall_hits(conversation_id, query=query, k=k)

    def teardown(self) -> None:
        """Nothing to delete: the alpha SDK has no forget/delete verb."""
        self._provenance.clear()

    def stats(self) -> dict:
        """Report how many facts this run stored, and the embedder if any."""
        stats: dict = {"stored_facts": len(self._provenance)}
        if self.embedding_model:
            stats["embedding_model"] = self.embedding_model
        return stats
