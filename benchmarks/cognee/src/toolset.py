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

"""Cognee's tool surface for `--mode agentic`.

The answering agent drives cognee's own `recall` over the conversation's
dataset; ingestion stays cognee's own `add` + `cognify`. The dataset is
the adapter's, never the agent's to choose.
"""

import time
from typing import Any

from amb.agent.toolset import SearchToolset
from amb.base import Memory
from src.memory import CogneeMemory


class CogneeSearchToolset(SearchToolset):
    """Cognee's search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose recall."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.recall, name="recall")

    @property
    def cognee(self) -> CogneeMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, CogneeMemory)
        return self.memory

    def recall(self, query: str) -> list[dict]:
        """Search the conversation's knowledge graph.

        Args:
            query: The information need as a natural phrase; matched
                against the stored text and the graph built from it.

        Returns:
            The best-matching remembered passages.
        """
        t0 = time.perf_counter()
        hits = self.cognee.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)
