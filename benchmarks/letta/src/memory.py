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

Workspace package ``letta-benchmark`` (this directory).
Needs a running Letta server (see benchmarks/letta/docker-compose.yaml). The
adapter benchmarks Letta's archival memory: one agent per conversation, turns
inserted as archival passages, search via passage retrieval.

Token accounting: letta spends inside its server, invisible to the harness's
in-process SDK wrappers, so the adapter computes it — every stored passage
and search query is tokenized with the embedding model's own tiktoken
encoding, and embeddings bill exactly their input, so the arithmetic equals
the wire. Calibrated 2026-08-11 against a counting reverse proxy on the
server's OpenAI traffic: identical to the token (53,735 over 1,000 passages
+ 189 queries; the proxy is retired, see git history). The counts assume one
embeddings call per passage insert / search and no server-side chunking —
both held for letta 0.16.8; recheck on version bumps.
"""

import os
from typing import ClassVar

from amb.base import Memory
from amb.callbacks import OpenAIUsageTracker
from amb.contracts import MemoryHit, Session

# write passages straight through the SDK; no LLM is invoked
INGEST_PASSAGES = "passages"
# hand the turns to the agent and let it decide what is worth keeping,
# which is the only mode where a letta run exercises its own model
INGEST_AGENT = "agent"
DEFAULT_INGEST = INGEST_PASSAGES


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

        The server requires a model handle at agent creation even though
        retrieval mode never invokes the LLM; the embedding model is what
        actually embeds and searches the archival passages. Both are
        letta handles (provider/name), settable via --param.
        """
        super().__init__(**params)
        self.model = model
        self.embedding_model = embedding_model
        # "passages" writes through the SDK and never invokes the LLM —
        # the surface every published letta run measured. "agent" hands
        # the turns to the agent, which decides what is worth keeping and
        # calls archival_memory_insert itself. Both are documented access
        # patterns; they measure different things, so they are different
        # rows. See `_ingest_via_agent`.
        self.ingest = ingest
        # where the server lives is infrastructure, so it comes from the
        # environment (compose sets it; localhost default for local runs)
        self.base_url = base_url or os.environ.get(
            "LETTA_BASE_URL", "http://localhost:8283"
        )
        self._agents: dict[str, str] = {}
        # passage id -> (turn_ids, session_id): scoring provenance, kept
        # local — the passages themselves carry only their text
        self._provenance: dict[str, tuple[list[str], str]] = {}
        # tokenizer-computed embedding spend (see module docstring), read
        # by TiktokenUsageTracker at the run's lifecycle boundaries
        self._usage = dict.fromkeys(OpenAIUsageTracker.KEYS, 0)

    def version(self) -> str | None:
        """The letta server's version — that is the system under test.

        Falls back to the client SDK's version when the server is not
        reachable at version-capture time.
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

    def _count_embedding(self, text: str) -> None:
        """Book one server-side embedding call for `text`."""
        self._usage["embedding_calls"] += 1
        self._usage["embedding_tokens"] += len(self._encoding.encode(text))

    def usage_counters(self) -> dict:
        """This instance's computed token spend so far (copy)."""
        return dict(self._usage)

    def _agent_id(self, conversation_id: str) -> str:
        if conversation_id not in self._agents:
            agent = self.client.agents.create(
                name=f"amb-{conversation_id}",
                memory_blocks=[],
                model=self.model,
                embedding=self.embedding_model,
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
        search uses to restore what a hit attests. The letta agent the
        passage lands in keeps every conversation's memory isolated.
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

        This is the mode where letta actually uses its LLM. The turns go
        in as user messages and the agent calls `archival_memory_insert`
        itself, which is the access pattern letta documents for agents,
        as against the `passages.*` endpoints it documents for
        developers.

        What that costs the measurement is worth stating. The agent
        decides what is worth keeping, so ingestion is *selective*: a
        later retrieval miss can be the search failing or the agent
        never having stored the fact, and this cannot tell them apart.
        The passages it writes are its own wording rather than the turn,
        so turn-level provenance is gone — the passages this session
        produced are recovered by diffing the agent's archive, which
        attests the session and nothing finer.

        The LLM spend happens inside letta's server and this adapter's
        tiktoken arithmetic models embeddings only, so a run in this mode
        under-reports its own cost.
        """
        agent_id = self._agent_id(conversation_id)
        before = set(self._passages(agent_id))
        # Serial, and it must stay that way: letta documents that
        # "sending multiple concurrent requests to the same agent can
        # lead to undefined behavior". `--workers N` is safe because
        # every conversation gets its own agent, so the parallelism is
        # across agents; sending a session's turns concurrently to speed
        # this up would put it back inside one.
        for turn in session.turns:
            self.client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": f"{turn.speaker}: {turn.text}"}],
            )
        for passage_id, text in self._passages(agent_id).items():
            if passage_id in before:
                continue
            # session only: the agent wrote its own wording, so there is
            # no turn to attribute and no text to match one against
            self._provenance[passage_id] = ([], session.session_id)
            # the passage the agent wrote is what letta embedded, so this
            # is the same exact arithmetic the write path uses — it is
            # just discovered after the insert rather than before it
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

        `passages.search` is the embedding-based endpoint (the same one the
        agent's own archival_memory_search tool uses); `passages.list` with
        its `search=` argument is literal text matching and returns noise
        for natural-language questions.
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
                    content=getattr(p, "text", ""),
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
