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

from amb.base import Memory
from amb.contracts import MemoryHit, QAPair, Sample, Session


class SearchToolset(FunctionToolset):
    """One conversation's memory, exposed to the answering agent as tools.

    Ships no tools of its own — subclasses add them with `add_function` and
    report every search through `record`. Created fresh per question.
    Everything the agent retrieves is recorded (`retrieved`, `num_searches`,
    `search_s`) so the harness can score retrieval and latency even though
    it no longer performs the searches.
    """

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation's memory."""
        super().__init__(**kwargs)
        self.memory = memory
        self.conversation_id = conversation_id
        self.k = k
        self.retrieved: list[MemoryHit] = []
        self.num_searches = 0
        self.search_s = 0.0
        self.callbacks = CallbackList([])
        self.sample: Sample | None = None
        self.qa: QAPair | None = None

    def observe(
        self, callbacks: CallbackList, sample: Sample, qa: QAPair
    ) -> None:
        """Bind the run's callbacks, so each search the agent runs is seen.

        The Runner calls this on the toolset an integration built: the
        agent's searches are the harness's only view of retrieval in
        agentic mode, and they must report through the same hooks as the
        searches it drives itself.
        """
        self.callbacks = callbacks
        self.sample = sample
        self.qa = qa

    def record(self, hits: list[MemoryHit], seconds: float) -> list[dict]:
        """Book one search's results and render them as a tool result.

        Provenance ids stay out of the tool result: they are the ground-truth
        labels retrieval is scored against.
        """
        self.search_s += seconds
        self.num_searches += 1
        self.retrieved.extend(hits)
        # unbound (a toolset built directly, in a test) means no observers
        if self.sample is not None and self.qa is not None:
            self.callbacks.on_search(self.sample, self.qa, hits, seconds)
        return [{"content": h.content, "score": h.score} for h in hits]


class MemoryToolset(SearchToolset):
    """The generic search surface: one `search_memory` tool.

    A BM25-style store has exactly one meaningful verb, so this doubles as
    the naive baseline's own toolset; any system without a richer surface
    can reuse it.
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


class IngestToolset(FunctionToolset):
    """One session's write surface, exposed to the ingesting agent as tools.

    Ships no tools of its own — integrations subclass it, add their native
    write tools with `add_function`, and report every write through
    `record_write`. Created fresh per session.

    The tools must look like real-time usage. An agent storing memories live
    knows nothing about conversation or session ids, so those never appear
    in a tool signature — the toolset is bound to the session being ingested
    and implementations attribute that provenance themselves (memory is
    conversation-bound, so how an adapter scopes and tags what it stores is
    its own business). Turn markers are the one citation handle tools may
    ask for: the turns are in a live agent's context.
    """

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session of one conversation."""
        super().__init__(**kwargs)
        self.memory = memory
        self.conversation_id = conversation_id
        self.session = session
        self.num_writes = 0
        self.write_s = 0.0
        self.callbacks = CallbackList([])
        self.sample: Sample | None = None

    def observe(self, callbacks: CallbackList, sample: Sample) -> None:
        """Bind the run's callbacks, so each write the agent makes is seen."""
        self.callbacks = callbacks
        self.sample = sample

    def record_write(self, seconds: float) -> None:
        """Book one write's timing."""
        self.write_s += seconds
        self.num_writes += 1
        if self.sample is not None:  # unbound means no observers
            self.callbacks.on_write(self.sample, self.session, seconds)

    def turn_ids(self) -> set[str]:
        """The session's valid turn markers, for validating agent citations."""
        return {turn.turn_id for turn in self.session.turns}
