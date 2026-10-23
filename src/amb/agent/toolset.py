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

import time
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

from amb.base import Callbacks, Memory
from amb.contracts import MemoryHit, QAPair, Sample


class SearchToolset(FunctionToolset):
    """One conversation's memory, exposed to the answering agent as tools.

    Ships no tools of its own — subclasses add them with `add_function` and
    report every search through `record`, so the harness still sees each
    search it no longer performs. Created fresh per question.
    """

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation's memory."""
        super().__init__(**kwargs)
        self.memory = memory
        self.conversation_id = conversation_id
        self.k = k
        self.num_hits = 0
        self.num_searches = 0
        self.search_s = 0.0
        self.callbacks = Callbacks()
        self.sample: Sample | None = None
        self.qa: QAPair | None = None

    def observe(self, callbacks: Callbacks, sample: Sample, qa: QAPair) -> None:
        """Bind the run's callbacks, so each search the agent runs is seen.

        The agent's searches are the harness's only view of the memory, so
        each one reports through the run's callback hooks.
        """
        self.callbacks = callbacks
        self.sample = sample
        self.qa = qa

    def record(self, hits: list[MemoryHit], seconds: float) -> list[dict]:
        """Book one search's results and render them as a tool result.

        Provenance ids stay out of the result: the agent sees only content and score.
        """
        self.search_s += seconds
        self.num_searches += 1
        self.num_hits += len(hits)
        # unbound (a toolset built directly, in a test) means no observers
        if self.sample is not None and self.qa is not None:
            self.callbacks.on_search(self.sample, self.qa, hits, seconds)
        return [{"content": h.content, "score": h.score} for h in hits]


class MemoryToolset(SearchToolset):
    """The generic search surface: one `search_memory` tool.

    Doubles as the naive baseline's toolset; any system without a richer
    surface can reuse it.
    """

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation's memory."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.search_memory, name="search_memory")

    def search_memory(self, query: str) -> list[dict]:
        """Search the user's long-term conversation memory.

        Args:
            query: Natural-language description of the information to find.
        """
        t0 = time.perf_counter()
        hits = self.memory.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)
