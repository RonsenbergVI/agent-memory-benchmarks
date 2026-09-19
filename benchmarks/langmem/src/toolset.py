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

"""LangMem's tool surface for `--mode agentic`.

These are LangMem's own two agent-facing verbs — `manage_memory` writes a
memory verbatim, `search_memory` searches what was written — with the
same bodies its `create_manage_memory_tool` and `create_search_memory_tool`
give an agent: a `put` and a `search` against the LangGraph store. The
namespace stays the adapter's, never the agent's to choose, so
conversation isolation cannot be argued away by a tool call.

Only the `create` action is exposed. LangMem's tool also takes `update`
and `delete`, both of which need a memory id the agent can only have got
from a search result — and ids are deliberately kept out of tool results
here, because they are retrieval's scoring labels.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import LangMemMemory


class LangMemSearchToolset(SearchToolset):
    """LangMem's search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose the search."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.search_memory, name="search_memory")

    @property
    def langmem(self) -> LangMemMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, LangMemMemory)
        return self.memory

    def search_memory(self, query: str) -> list[dict]:
        """Search your long-term memories for this conversation.

        Args:
            query: The information need as a natural phrase; matched
                against the stored memories semantically.

        Returns:
            The best-matching remembered memories.
        """
        t0 = time.perf_counter()
        hits = self.langmem.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class LangMemIngestToolset(IngestToolset):
    """LangMem's write path, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose the write."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.manage_memory, name="manage_memory")

    @property
    def langmem(self) -> LangMemMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, LangMemMemory)
        return self.memory

    def manage_memory(self, content: str, source_turn_ids: list[str]) -> str:
        """Create a memory holding what is worth remembering.

        Args:
            content: The memory, written as a self-contained statement —
                it is stored exactly as given, with no extraction pass.
            source_turn_ids: The ids of the turns it came from, exactly
                as shown in the transcript.

        Returns:
            Whether the memory was created.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.langmem.store_memory(
            self.conversation_id,
            content,
            session_id=self.session.session_id,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "created, but unknown turn ids were dropped"
        return "created"
