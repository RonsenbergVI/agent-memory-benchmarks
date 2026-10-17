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

The answering agent drives `search_memory`; ingestion stays mem0's own
LLM extraction. The adapter always sets the ``user_id`` scope, so
conversation isolation is never left to the agent.
"""

import time
from typing import Any

from amb.agent.toolset import SearchToolset
from amb.base import Memory


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
