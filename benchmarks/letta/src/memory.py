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

"""Letta (letta-ai/letta) — OS-style hierarchical agent memory.

Needs a running Letta server (benchmarks/letta/docker-compose.yaml): one agent
per conversation, turns inserted as archival passages, search via passage
retrieval.

Token accounting: letta spends inside its server, invisible to the harness's
in-process SDK wrappers, so the adapter computes it — every stored passage and
search query is tokenized with the embedding model's own tiktoken encoding.
Calibrated 2026-08-11 against a counting reverse proxy on the server's OpenAI
traffic: identical to the token (53,735 over 1,000 passages + 189 queries; the
proxy is retired, see git history). Assumes one embeddings call per passage
insert / search and no server-side chunking — both held for letta 0.16.8;
recheck on version bumps. Under ``--param ingest=agent`` the agent's own LLM
turns are billed from the usage letta reports on each response.
"""

import os
from typing import ClassVar

from amb.base import Memory
from amb.callbacks import OpenAIUsageTracker
from amb.contracts import MemoryHit, Session

INGEST_PASSAGES = "passages"  # SDK writes, no LLM invoked
INGEST_AGENT = "agent"  # the only mode where a letta run exercises its own model
DEFAULT_INGEST = INGEST_PASSAGES
ARCHIVAL_INSERT_TOOL = "archival_memory_insert"  # attached under `ingest=agent`


class LettaMemory(Memory):
    """Letta: OS-style hierarchical agent memory, benchmarked via archival passages."""

    name: ClassVar[str] = "letta"
    description: ClassVar[str] = "Letta — hierarchical (OS-style) agent memory"
    sdk_dist: ClassVar[str | None] = "letta-client"

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "openai/gpt-5-mini",
        embedding_model: str = "openai/text-embedding-3-small",
        ingest: str = DEFAULT_INGEST,
        **params: object,
    ) -> None:
        """Point the adapter at a Letta server and pin its agent models.

        The server requires an LLM handle at agent creation even though
        retrieval mode never invokes it; the embedding model is what actually
        embeds and searches. Both are letta handles (provider/name).
        """
        super().__init__(**params)
        self.model = model
        self.embedding_model = embedding_model
        # "passages" (SDK writes, what published runs measured) and "agent"
        # (see `_ingest_via_agent`) measure different things: different rows
        self.ingest = ingest
        # server location is infrastructure: from the environment (compose sets it)
        self.base_url = base_url or os.environ.get(
            "LETTA_BASE_URL", "http://localhost:8283"
        )
        self._agents: dict[str, str] = {}
        # passage id -> (turn_ids, session_id): local scoring provenance
        self._provenance: dict[str, tuple[list[str], str]] = {}
        # computed embedding spend (module docstring), read by TiktokenUsageTracker
        self._usage = dict.fromkeys(OpenAIUsageTracker.KEYS, 0)

    def version(self) -> str | None:
        """The letta server's version — that is the system under test.

        Falls back to the client SDK's version when the server is unreachable.
        """
        import json
        import urllib.request

        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/health/", timeout=5) as r:
                server = json.load(r).get("version")
            if server:
                return str(server)
        except OSError:
            pass
        return super().version()

    def setup(self) -> None:
        """Connect to the Letta server and pin the counting tokenizer."""
        import tiktoken
        from letta_client import Letta

        self.client = Letta(base_url=self.base_url)
        # letta handles are provider/name; tiktoken wants the bare name
        name = self.embedding_model.rpartition("/")[2]
        try:
            self._encoding = tiktoken.encoding_for_model(name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_reported_usage(self, response: object) -> None:
        """Book the LLM spend letta reports for one agent turn.

        Billed from letta's reported usage, not estimated. `reasoning_tokens`
        are already inside `completion_tokens` — deliberately not added again.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage["llm_calls"] += 1
        self._usage["llm_input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage["llm_output_tokens"] += getattr(usage, "completion_tokens", 0) or 0

    def _count_embedding(self, text: str) -> None:
        """Book one server-side embedding call for `text`."""
        self._usage["embedding_calls"] += 1
        self._usage["embedding_tokens"] += len(self._encoding.encode(text))

    def usage_counters(self) -> dict:
        """This instance's computed token spend so far (copy)."""
        return dict(self._usage)

    def _agent_id(self, conversation_id: str) -> str:
        if conversation_id not in self._agents:
            # `ingest=agent` needs the archival write verb attached: without it
            # the agent reaches for `memory_insert` (a core-memory block edit)
            # and the archive stays empty while the run looks healthy.
            tools = [ARCHIVAL_INSERT_TOOL] if self.ingest == INGEST_AGENT else None
            agent = self.client.agents.create(
                name=f"amb-{conversation_id}",
                memory_blocks=[],
                model=self.model,
                embedding=self.embedding_model,
                **({"tools": tools} if tools else {}),
            )
            self._agents[conversation_id] = agent.id
        return self._agents[conversation_id]

    def store(
        self,
        conversation_id: str,
        text: str,
        *,
        session_id: str,
        turn_ids: list[str],
    ) -> None:
        """Insert one archival passage inside the conversation's agent.

        `session_id` and `turn_ids` feed the local provenance map, which
        search uses to restore what a hit attests.
        """
        created = self.client.agents.passages.create(
            agent_id=self._agent_id(conversation_id), text=text
        )
        self._count_embedding(text)
        for passage in created if isinstance(created, list) else [created]:
            pid = getattr(passage, "id", None)
            if pid is not None:
                self._provenance[str(pid)] = (list(turn_ids), session_id)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Store the session, by the configured ingestion mode."""
        if self.ingest == INGEST_AGENT:
            self._ingest_via_agent(conversation_id, session)
            return
        for turn in session.turns:
            self.store(
                conversation_id,
                f"{turn.speaker}: {turn.text}",
                session_id=session.session_id,
                turn_ids=[turn.turn_id],
            )

    def _ingest_via_agent(self, conversation_id: str, session: Session) -> None:
        """Hand the turns to the agent and let it write what it keeps.

        The mode where letta actually uses its LLM: turns go in as user
        messages and the agent calls `archival_memory_insert` itself (letta's
        documented agent access pattern, vs the developer `passages.*`
        endpoints). Ingestion is *selective* — a retrieval miss can be search
        failing or the agent never storing the fact, indistinguishably — and
        the agent writes its own wording, so provenance is session-level only,
        recovered by diffing the archive. Cost stays fully accounted: LLM
        usage is billed from letta's responses, passages tokenized as usual.
        """
        agent_id = self._agent_id(conversation_id)
        before = set(self._passages(agent_id))
        # Serial by requirement: letta documents that "sending multiple
        # concurrent requests to the same agent can lead to undefined
        # behavior". `--workers N` is safe — that parallelism is across agents.
        for turn in session.turns:
            response = self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": f"{turn.speaker}: {turn.text}"}],
            )
            self._count_reported_usage(response)
        for passage_id, text in self._passages(agent_id).items():
            if passage_id in before:
                continue
            # agent wording: no turn to attribute, session-level provenance only
            self._provenance[passage_id] = ([], session.session_id)
            # letta embedded the agent's passage — same arithmetic as the write path
            self._count_embedding(text)

    def _passages(self, agent_id: str) -> dict[str, str]:
        """The agent's archive right now, as passage id -> text."""
        response = self.client.agents.passages.list(agent_id=agent_id)
        passages = getattr(response, "results", response) or []
        return {
            str(p.id): getattr(p, "text", "") or ""
            for p in passages
            if getattr(p, "id", None)
        }

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best archival passages by semantic search.

        `passages.search` is the embedding-based endpoint (what the agent's own
        archival_memory_search tool uses); `passages.list(search=)` is literal
        text matching and returns noise for natural-language questions.
        """
        agent_id = self._agent_id(conversation_id)
        response = self.client.agents.passages.search(
            agent_id=agent_id, query=query, top_k=k
        )
        self._count_embedding(query)
        passages = getattr(response, "results", response) or []
        hits = []
        for p in passages:
            turn_ids, session_id = self._provenance.get(str(p.id), ([], None))
            hits.append(
                MemoryHit(
                    # search Results carry `content`, list passages carry `text`;
                    # reading only `text` handed `--model` runs empty hits.
                    content=getattr(p, "content", None) or getattr(p, "text", "") or "",
                    turn_ids=list(turn_ids),
                    session_ids=[session_id] if session_id else [],
                    metadata={"id": str(p.id)},
                )
            )
        return hits

    def teardown(self) -> None:
        """Delete every agent this run created."""
        for agent_id in self._agents.values():
            try:
                self.client.agents.delete(agent_id)
            except Exception:
                pass
        self._agents.clear()
        self._provenance.clear()

    def stats(self) -> dict:
        """Report how this run ingested."""
        return {"ingest": self.ingest}
