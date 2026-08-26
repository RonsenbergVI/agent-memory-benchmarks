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

"""agentmemory's tool surface for `--mode agentic`.

The agent drives agentmemory's own verbs: `observe` records something
that happened, `recall` runs the two-step smart search over what was
recorded. Those are the verbs the project's own hooks and MCP tools are
built on, so an agent using them is using agentmemory the way it is meant
to be used.

Which session and which agent id an observation is filed under stay the
adapter's, never the agent's to choose, so conversation isolation cannot
be argued away by a tool call.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import AgentMemoryMemory


class AgentMemorySearchToolset(SearchToolset):
    """agentmemory's smart search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose recall."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.recall, name="recall")

    @property
    def agentmemory(self) -> AgentMemoryMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, AgentMemoryMemory)
        return self.memory

    def recall(self, query: str) -> list[dict]:
        """Search everything observed in this conversation.

        Args:
            query: The information need as a natural phrase; matched by
                keyword and by meaning against what was observed.

        Returns:
            The best-matching observations.
        """
        t0 = time.perf_counter()
        hits = self.agentmemory.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class AgentMemoryIngestToolset(IngestToolset):
    """agentmemory's write path, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose observe."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.observe, name="observe")

    @property
    def agentmemory(self) -> AgentMemoryMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, AgentMemoryMemory)
        return self.memory

    def observe(self, content: str, source_turn_ids: list[str]) -> str:
        """Record one observation from this conversation.

        Args:
            content: What was said or decided, in the words it happened
                in — a keyless agentmemory stores this verbatim and does
                not rewrite it.
            source_turn_ids: The ids of the turns it came from, exactly
                as shown in the transcript.

        Returns:
            A status message indicating whether the observation was recorded.
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not recorded: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.agentmemory.observe(
            self.conversation_id,
            content,
            session_id=self.session.session_id,
            timestamp=self.session.timestamp,
            # one observation cites one turn in the store's own map; the
            # first is the anchor, and the rest stay in the tool's answer
            turn_id=cited[0],
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "recorded, but unknown turn ids were dropped from the citation"
        return "recorded"
