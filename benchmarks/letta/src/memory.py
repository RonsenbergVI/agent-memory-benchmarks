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

import json
import os
from typing import ClassVar

from amb.base import Memory
from amb.callbacks import OpenAIUsageTracker
from amb.contracts import MemoryHit, Session

# retrieve straight from the archival store; no LLM is invoked
RECALL_PASSAGES = "passages"
# ask the agent, so letta's own model formulates the query and reads the
# results — the only mode in which a letta run exercises its LLM
RECALL_AGENT = "agent"
DEFAULT_RECALL = RECALL_PASSAGES
# the tool letta's agent calls to reach its archival store
ARCHIVAL_SEARCH_TOOL = "archival_memory_search"


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
        recall: str = DEFAULT_RECALL,
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
        # "passages" retrieves straight from the archival store and never
        # invokes the LLM — the surface every published letta run measured.
        # "agent" asks the agent instead, so letta's own model chooses the
        # query and reads the results back. Different subject, different
        # row; see `_search_via_agent`.
        self.recall = recall
        # where the server lives is infrastructure, so it comes from the
        # environment (compose sets it; localhost default for local runs)
        self.base_url = base_url or os.environ.get(
            "LETTA_BASE_URL", "http://localhost:8283"
        )
        self._agents: dict[str, str] = {}
        # passage id -> (turn_ids, session_id): scoring provenance, kept
        # local — the passages themselves carry only their text
        self._provenance: dict[str, tuple[list[str], str]] = {}
        # passage text -> the same, for `recall=agent`: the agent's tool
        # return renders passages as text and promises no id to match on
        self._provenance_by_text: dict[str, tuple[list[str], str]] = {}
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
        self._provenance_by_text[text.strip()] = (list(turn_ids), session_id)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Insert each turn of the session as an archival passage."""
        for turn in session.turns:
            self.store(
                conversation_id,
                f"{turn.speaker}: {turn.text}",
                session_id=session.session_id,
                turn_ids=[turn.turn_id],
            )

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best archival passages, by the configured recall mode."""
        if self.recall == RECALL_AGENT:
            return self._search_via_agent(conversation_id, query, k)
        return self._search_via_passages(conversation_id, query, k)

    def _search_via_agent(
        self, conversation_id: str, query: str, k: int = 10
    ) -> list[MemoryHit]:
        """Ask the agent, and take what its own archival search returned.

        This is the mode where letta actually uses its LLM: the question
        goes to the agent, the agent decides how to search, and the
        passages come back out of its `archival_memory_search` tool
        returns. What is measured is therefore letta's retrieval *plus*
        its model's query formulation — strictly more than
        `_search_via_passages` measures, and not comparable with it.

        The LLM spend lives inside letta's server and is not counted by
        the tiktoken arithmetic in this adapter, which models embeddings
        only. A run in this mode under-reports its own cost.
        """
        agent_id = self._agent_id(conversation_id)
        response = self.client.agents.messages.create(
            agent_id=agent_id,
            messages=[{"role": "user", "content": query}],
        )
        hits: list[MemoryHit] = []
        for message in getattr(response, "messages", None) or []:
            if getattr(message, "message_type", None) != "tool_return_message":
                continue
            if getattr(message, "name", None) != ARCHIVAL_SEARCH_TOOL:
                continue
            if getattr(message, "is_err", False):
                continue
            hits.extend(self._hits_from_tool_return(message))
        return hits[:k]

    def _hits_from_tool_return(self, message: object) -> list[MemoryHit]:
        """Passages named by one archival_memory_search return.

        The return is the tool's own rendering, so provenance is restored
        by matching the text back to what was inserted rather than by an
        id the tool does not promise to carry.
        """
        raw = getattr(message, "tool_return", None)
        if raw is None:
            return []
        texts: list[str] = []
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = raw
        else:
            parsed = raw
        if isinstance(parsed, str):
            texts = [parsed]
        elif isinstance(parsed, dict):
            items = parsed.get("results") or parsed.get("passages") or []
            texts = [self._text_of(i) for i in items]
        elif isinstance(parsed, list):
            texts = [self._text_of(i) for i in parsed]
        hits = []
        for text in [t for t in texts if t]:
            turn_ids, session_id = self._provenance_by_text.get(text, ([], None))
            hits.append(
                MemoryHit(
                    content=text,
                    turn_ids=list(turn_ids),
                    session_ids=[session_id] if session_id else [],
                    metadata={"via": ARCHIVAL_SEARCH_TOOL},
                )
            )
        return hits

    @staticmethod
    def _text_of(item: object) -> str:
        """The passage text out of whatever shape the tool rendered."""
        if isinstance(item, str):
            return item.strip()
        if isinstance(item, dict):
            for key in ("text", "content", "passage"):
                if value := item.get(key):
                    return str(value).strip()
        return ""

    def _search_via_passages(
        self, conversation_id: str, query: str, k: int = 10
    ) -> list[MemoryHit]:
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
        self._provenance_by_text.clear()

    def stats(self) -> dict:
        """Report how this run recalled."""
        return {"recall": self.recall}
