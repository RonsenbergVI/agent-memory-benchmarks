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

"""Supermemory's tool surface for `--mode agentic`.

The agent drives Supermemory's own verbs: `remember` adds a document and
lets the service extract the memories from it, `search_memory` searches
what it extracted. The container tag stays the adapter's, never the
agent's to choose, so conversation isolation cannot be argued away by a
tool call.

`remember` blocks until the document is searchable. That is slower than
the write the agent thinks it is making, and it is the honest shape: an
agent told "stored" about a document that is still queued would go on to
search for it and find nothing.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import SupermemoryMemory


class SupermemorySearchToolset(SearchToolset):
    """Supermemory's search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose search."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.search_memory, name="search_memory")

    @property
    def supermemory(self) -> SupermemoryMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, SupermemoryMemory)
        return self.memory

    def search_memory(self, query: str) -> list[dict]:
        """Search the memories extracted from this conversation.

        Args:
            query: The information need as a natural phrase; matched
                against the memories, not the raw transcript.

        Returns:
            The best-matching remembered facts.
        """
        t0 = time.perf_counter()
        hits = self.supermemory.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class SupermemoryIngestToolset(IngestToolset):
    """Supermemory's write path, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose remember."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.remember, name="remember")

    @property
    def supermemory(self) -> SupermemoryMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, SupermemoryMemory)
        return self.memory

    def remember(self, content: str, source_turn_ids: list[str]) -> str:
        """Hand one piece of content to Supermemory to remember.

        Args:
            content: What is worth remembering, phrased so the service's
                own extraction has something to work with — a statement,
                not a keyword.
            source_turn_ids: The ids of the turns the content came from,
                exactly as shown in the transcript.

        Returns:
            Whether the content was stored.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.supermemory.store(
            self.conversation_id,
            content,
            session_id=self.session.session_id,
            turn_ids=cited,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "stored, but unknown turn ids were dropped from the citation"
        return "stored"
