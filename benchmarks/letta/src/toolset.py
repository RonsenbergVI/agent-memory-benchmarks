# MIT License

# Copyright (c) 2026 René-Jean Corneille

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Letta's native tool surface for `--mode agentic`.

Mirrors the archival-memory verbs a letta agent has in real time:
`archival_memory_insert` at ingestion, `archival_memory_search` at search.
The conversation's letta agent is the isolation scope, chosen by the
adapter — never by the model.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import LettaMemory


class LettaSearchToolset(SearchToolset):
    """Letta's archival passage search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose search."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.archival_memory_search, name="archival_memory_search")

    def archival_memory_search(self, query: str) -> list[dict]:
        """Search archival memory semantically.

        Args:
            query: Natural-language description of the information to find.

        Returns:
            The best-matching archival passages with their scores.
        """
        t0 = time.perf_counter()
        hits = self.memory.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class LettaIngestToolset(IngestToolset):
    """Letta's archival passage insert, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose the insert tool."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.archival_memory_insert, name="archival_memory_insert")

    @property
    def letta(self) -> LettaMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, LettaMemory)
        return self.memory

    def archival_memory_insert(self, content: str, source_turn_ids: list[str]) -> str:
        """Write one memory to archival storage.

        Args:
            content: One self-contained statement worth remembering, phrased
                so a later semantic search can find it.
            source_turn_ids: The turn markers (left column of the transcript)
                the memory came from.

        Returns:
            Whether the memory was stored.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn markers exist in this session"
        t0 = time.perf_counter()
        self.letta.store(
            self.conversation_id,
            f"{self.letta.session_header(self.session)} {content}",
            session_id=self.session.session_id,
            turn_ids=cited,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "stored, but unknown turn markers were dropped from the citation"
        return "stored"
