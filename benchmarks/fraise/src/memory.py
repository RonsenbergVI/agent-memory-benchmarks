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

Workspace package ``fraise-benchmark`` (this directory).
Needs a running fraise server (see benchmarks/fraise/docker-compose.yaml);
FRAISE_BASE_URL names it, defaulting to localhost. The fraise SDK has no
extractor of its own, so direct-mode ingestion runs one here: an extraction
step (gpt-5-mini by default, the same model every other system in the
comparison ingests with) distills each session into standalone facts stored
with their entities, and the SDK's OpenAIEmbedder (text-embedding-3-small
by default) encodes every fact and query in-process through the openai SDK,
so the token spend is visible to the OpenAIUsageTracker. In agentic mode
the driving agent is the extractor (through the write toolset), and the
ingestion model here plays no part.

``--param model=none`` restores the raw one-fact-per-turn ingestion;
``--param embedding_model=none`` drops the embedder — with both unset,
fraise runs its out-of-the-box hybrid retrieval (full-text + graph walk)
with no LLM or embedding calls at all.

Each conversation is isolated by a `conv-<id>` topic anchor — the FQL
grammar makes every anchor a hard filter, so recall never crosses
conversations. A local value -> (turns, session) map restores provenance
on recall, since hits return only value/score/timestamp. The alpha SDK has
no delete, so teardown is a no-op — the compose stack runs the server
without a volume and each `up` starts empty.
"""

import json
import os
import re
from typing import TYPE_CHECKING, ClassVar, cast

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.completion_create_params import ResponseFormat

# gpt-5-mini is a reasoning model: its hidden reasoning tokens draw from the
# same completion budget as the visible JSON, and reasoning length is not
# deterministic call to call. Without a cap, an unlucky combination of a
# long reasoning pass and a fact-dense session truncates the response
# mid-string — observed in practice (json.JSONDecodeError on an otherwise
# identical, previously-successful session). This cap gives generous
# headroom over what a single session's extraction has needed so far
# (~2.7k total tokens, ~2k of it reasoning, on the sessions measured).
EXTRACTION_MAX_COMPLETION_TOKENS = 16000

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# every recall walks the graph two hops from a match by default
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

# strict structured-output schema for one session's extracted memories
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
        self.base_url = base_url or os.environ.get(
            "FRAISE_BASE_URL", "http://localhost:9876"
        )
        # explicit none (--param model=none) restores raw per-turn ingestion
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings; the SDK wants an int or None
        self.embedding_dimensions = (
            int(embedding_dimensions) if embedding_dimensions else None
        )
        # graph-walk depth on recall, 2 by default (--param depth=N);
        # an explicit --param depth=none restores the server's own default,
        # same idiom as the model params above
        self.depth = int(depth) if depth else None
        # (conversation, stored value) -> (turn_ids, session_id): hits return
        # only the value, so provenance is restored from what we stored
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

    @staticmethod
    def _tokens(values: list[str] | None) -> list[str]:
        """Free-form topics/entities as FQL tokens.

        The grammar splits on whitespace, so a multi-word entity ("New
        York", extracted or agent-supplied) must become one hyphenated
        token to survive as a single filter.
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

        The anchors keep every conversation's recall isolated; extra topics
        and entities (agentic mode: the agent's choice) ride along. `turn_ids`
        feed the local provenance map, which search uses to restore what a
        hit attests.
        """
        # FQL phrases have no escape for an apostrophe, so swap it for
        # the typographic one; the value round-trips identically, which
        # the provenance map relies on
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
            # top" on content our token-only client validation doesn't
            # catch) — log the exact payload so the offending value/topic/
            # entity is diagnosable instead of just the server's column
            # offset into a query string we never see
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

        With an ingestion model, one LLM call distills the session into
        standalone facts — the fraise SDK has no extractor, so the adapter
        supplies the step every other system runs inside its SDK or server.
        The cited turn markers become each fact's provenance; entities feed
        the server's graph walk.
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

        A truncated/malformed response (see `EXTRACTION_MAX_COMPLETION_TOKENS`)
        is logged and treated as no facts rather than raised — matching mem0's
        own extractor, which hits the same class of malformed response from
        the same family of models and skips it rather than failing the run.
        """
        timestamp = f" of {session.timestamp}" if session.timestamp else ""
        transcript = f"Conversation{timestamp}\n" + "\n".join(
            f"{turn.turn_id} | {turn.speaker}: {turn.text}" for turn in session.turns
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        # the literal is a valid ResponseFormatJSONSchema; the cast bridges
        # the nested-schema dict the checker cannot narrow into the TypedDict
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
        # setup() builds self._openai only when a model is set, so the
        # extraction path implies one
        assert self.model is not None
        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=EXTRACTION_MAX_COMPLETION_TOKENS,
            response_format=response_format,
        )
        choice = response.choices[0]
        # a refusal or empty completion has content=None — the same class of
        # malformed response as unparseable JSON, so it takes the same exit
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

        `query` travels as one quoted phrase term — literal inside its
        quotes, so it can never collide with the grammar's reserved words —
        and doubles as the embed text when an embedder is configured. No
        bare keyword terms are ever sent. `self.depth` bounds how many
        graph hops a match may walk to pull in related facts — 2 by
        default, `--param depth=N` to change it, `--param depth=none` for
        the server's own default.
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
            # same diagnostic idiom as store(): log the exact inputs so a
            # rejected query is diagnosable from the benchmark log alone,
            # which outlives the server container and its logs
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
        """Return the k best facts for the query, inside the conversation.

        The question goes over as the quoted phrase term only — never
        tokenized into bare keywords, where grammar words like "topic",
        "since" or "remember" would break the server's parse.
        """
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
