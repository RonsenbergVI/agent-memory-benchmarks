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

import threading
import time

from amb.base.callback import Callback
from amb.base.memory import Memory
from amb.contracts import IngestionRecord, MemoryHit, QAPair, Sample


class TimingTracker(Callback):
    """Times every phase of a run — the harness's own core measurement.

    Writes `ingest_s` on sample stats, `conversation_s` on its
    record, and `search_s`/`answer_s`/`num_searches` on every question row. Attached by
    `Runner.core_callback_classes`, so a run cannot be missing latencies.

    State is keyed by sample, not thread: the agent's searches arrive from
    pydantic-ai's own worker thread, so thread-local marks would lose
    exactly the events a run exists to measure. A run never
    has two samples of the same id in flight.
    """

    def __init__(self) -> None:
        """Start with no sample being timed."""
        self._samples: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _state(self, sample: Sample) -> dict:
        """This sample's clocks and accumulators, created on first use."""
        with self._lock:
            return self._samples.setdefault(sample.sample_id, {})

    def _mark(self, sample: Sample, phase: str) -> None:
        self._state(sample)[f"mark:{phase}"] = time.perf_counter()

    def _elapsed(self, sample: Sample, phase: str) -> float | None:
        """Seconds since the phase was marked, or None if it never began."""
        mark = self._state(sample).pop(f"mark:{phase}", None)
        return None if mark is None else time.perf_counter() - mark

    def on_sample_begin(self, sample: Sample, system: Memory) -> None:
        """Start this sample's clocks."""
        with self._lock:
            self._samples[sample.sample_id] = {}

    def on_sample_end(self, sample: Sample, system: Memory) -> None:
        """Drop this sample's clocks."""
        with self._lock:
            self._samples.pop(sample.sample_id, None)

    def on_ingest_begin(self, sample: Sample, system: Memory) -> None:
        """Start the ingestion clock."""
        self._mark(sample, "ingest")

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """Book ingestion's time; a run that ingested nothing charges zero."""
        stats["ingest_s"] = self._elapsed(sample, "ingest") or 0.0

    def on_conversation_begin(self, sample: Sample) -> None:
        """Start the question-phase clock."""
        self._mark(sample, "conversation")

    def on_conversation_end(self, sample: Sample, stats: IngestionRecord) -> None:
        """Book the question phase's time on the sample's record."""
        stats.conversation_s = self._elapsed(sample, "conversation")

    def on_question_begin(self, sample: Sample, qa: QAPair) -> None:
        """Reset the per-question accumulators."""
        self._state(sample).update(search_s=0.0, num_searches=0)

    def on_search(
        self,
        sample: Sample,
        qa: QAPair,
        hits: list[MemoryHit],
        seconds: float,
    ) -> None:
        """Accumulate one search, whoever drove it and timed it."""
        state = self._state(sample)
        state["search_s"] = state.get("search_s", 0.0) + seconds
        state["num_searches"] = state.get("num_searches", 0) + 1

    def on_answer_begin(self, sample: Sample, qa: QAPair) -> None:
        """Start the answer clock."""
        self._mark(sample, "answer")

    def on_answer_end(self, sample: Sample, qa: QAPair) -> None:
        """Stop the answer clock, holding the value for the row."""
        self._state(sample)["answer_s"] = self._elapsed(sample, "answer")

    def on_question_end(self, sample: Sample, qa: QAPair, row: dict) -> None:
        """Book the question's latencies on its row."""
        state = self._state(sample)
        row["search_s"] = state.get("search_s", 0.0)
        row["num_searches"] = state.get("num_searches", 0)
        answer_s = state.pop("answer_s", None)
        if answer_s is not None:
            row["answer_s"] = answer_s
