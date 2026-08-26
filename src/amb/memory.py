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

"""Naive lexical baseline: stores raw turns, retrieves with BM25.

The floor every real memory system should beat. No LLM calls, no external
services, no persistence — also serves as the reference plugin implementation.
"""

import math
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any, ClassVar

from amb.agent.toolset import IngestToolset, MemoryToolset
from amb.base import Benchmark, Memory
from amb.contracts import MemoryHit, Session

_TOKEN = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class NaiveMemory(Memory):
    """BM25 over raw turns: the baseline every real system must beat."""

    name: ClassVar[str] = "naive"
    description: ClassVar[str] = "BM25 over raw turns; the baseline to beat"
    sdk_dist: ClassVar[str | None] = "amb"

    def __init__(self, k1: float = 1.5, b: float = 0.75, **params: object) -> None:
        """Set the BM25 term-saturation (k1) and length-normalization (b)."""
        super().__init__(k1=k1, b=b, **params)
        self.k1, self.b = k1, b
        # conversation_id -> list of (turn_ids, session_id, text, tf)
        self._docs: dict[str, list] = defaultdict(list)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Index every turn of the session for this conversation."""
        for turn in session.turns:
            text = f"{turn.speaker}: {turn.text}"
            if session.timestamp:
                text = f"({session.timestamp}) {text}"
            self._docs[conversation_id].append(
                (
                    [turn.turn_id],
                    session.session_id,
                    text,
                    Counter(_tokens(f"{turn.speaker} {turn.text}")),
                )
            )

    def store(
        self,
        conversation_id: str,
        content: str,
        turn_ids: list[str],
        session_id: str,
    ) -> None:
        """Index one agent-authored memory with its provenance (agentic mode)."""
        self._docs[conversation_id].append(
            (list(turn_ids), session_id, content, Counter(_tokens(content)))
        )

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best-scoring turns for the query, BM25-ranked."""
        docs = self._docs.get(conversation_id, [])
        if not docs:
            return []
        n = len(docs)
        df = Counter()
        for *_, tf in docs:
            df.update(tf.keys())
        avgdl = sum(sum(tf.values()) for *_, tf in docs) / n

        scored = []
        for turn_ids, session_id, text, tf in docs:
            dl = sum(tf.values()) or 1
            score = 0.0
            for term in _tokens(query):
                if term not in tf:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += (
                    idf
                    * tf[term]
                    * (self.k1 + 1)
                    / (tf[term] + self.k1 * (1 - self.b + self.b * dl / avgdl))
                )
            if score > 0:
                scored.append((score, turn_ids, session_id, text))

        scored.sort(key=lambda s: -s[0])
        return [
            MemoryHit(
                content=text,
                score=score,
                turn_ids=turn_ids,
                session_ids=[session_id],
            )
            for score, turn_ids, session_id, text in scored[:k]
        ]

    def teardown(self) -> None:
        """Drop every indexed turn."""
        self._docs.clear()

    def stats(self) -> dict:
        """Report how many turns are stored and roughly how much memory."""
        turns = sum(len(d) for d in self._docs.values())
        return {"stored_turns": turns, "approx_bytes": sys.getsizeof(self._docs)}


class NaiveIngestToolset(IngestToolset):
    """The naive store's write surface: index one memory at a time."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose the write tool."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.store_memory, name="store_memory")

    @property
    def naive(self) -> NaiveMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, NaiveMemory)
        return self.memory

    def store_memory(self, content: str, source_turn_ids: list[str]) -> str:
        """Store one memory about this session for later retrieval.

        Args:
            content: The fact to remember, phrased so a search can find it.
            source_turn_ids: The ids of the turns the fact came from, exactly
                as shown in the transcript.

        Returns:
            Whether the memory was stored.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.naive.store(self.conversation_id, content, cited, self.session.session_id)
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "stored, but unknown turn ids were dropped from the citation"
        return "stored"


class NaiveBenchmark(Benchmark):
    """The benchmark object the `naive` entry point resolves to."""

    name: ClassVar[str] = "naive"
    system_class = NaiveMemory
    # the baseline's surface is exactly the generic pair: one search, one write tool
    search_toolset_class = MemoryToolset
    ingest_toolset_class = NaiveIngestToolset
