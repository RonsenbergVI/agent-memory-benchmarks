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

"""Mem0's native tool surface for `--mode agentic`.

The agent is the extractor here: it writes distilled facts through
`add_memory` (stored verbatim, ``infer=False``) and retrieves through
mem0's semantic `search_memory`. Conversation isolation stays forced — the
``user_id`` scope is set by the adapter, never left to the agent.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import Mem0Memory


class Mem0SearchToolset(SearchToolset):
    """Mem0's semantic search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose search."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.search_memory, name="search_memory")

    def search_memory(self, query: str) -> list[dict]:
        """Search the stored memories semantically.

        Args:
            query: Natural-language description of the information to find.

        Returns:
            The best-matching memories with their scores.
        """
        t0 = time.perf_counter()
        hits = self.memory.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class Mem0IngestToolset(IngestToolset):
    """Mem0's write surface, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose the write tool."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.add_memory, name="add_memory")

    @property
    def mem0(self) -> Mem0Memory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, Mem0Memory)
        return self.memory

    def add_memory(self, fact: str, source_turn_ids: list[str]) -> str:
        """Store one memory, exactly as written.

        Args:
            fact: One self-contained statement worth remembering, phrased so
                a later semantic search can find it.
            source_turn_ids: The ids of the turns the fact came from, exactly
                as shown in the transcript.

        Returns:
            Whether the memory was stored.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.mem0.store(
            self.conversation_id,
            fact,
            session_id=self.session.session_id,
            turn_ids=cited,
            timestamp=self.session.timestamp,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "stored, but unknown turn ids were dropped from the citation"
        return "stored"
