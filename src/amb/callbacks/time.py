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
from typing import TYPE_CHECKING

from amb.base.callback import Callback
from amb.base.memory import Memory
from amb.contracts import IngestionRecord, MemoryHit, QAPair, Run, Sample, Session

if TYPE_CHECKING:
    from amb.runner import RunConfig


class TimingTracker(Callback):
    """Times every phase of a run — the harness's own core measurement.

    Produces `ingest_s` and `write_s` on each sample's stats,
    `conversation_s` on its record, and `search_s` / `answer_s` (plus
    `num_searches`, where an agent chose how often to search) on every
    question row. Attached by `Runner.core_callback_classes` ahead of the
    benchmark's own callbacks, so a run cannot be missing its latencies.

    State is keyed by sample, not thread: `--workers` runs samples
    concurrently, and in agentic mode the searches and writes are reported
    from pydantic-ai's own worker thread, so thread-local marks would lose
    exactly the events this mode exists to measure. Every hook carries its
    sample, and a run never has two samples of the same id in flight.
    """

    def __init__(self) -> None:
        """Start with no sample being timed."""
        self._samples: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._agentic = False

    def on_run_begin(self, config: "RunConfig", run: Run) -> None:
        """Note the mode: it decides whether search counts are observations.

        In direct mode the harness searches exactly once per question, so
        recording that is bookkeeping, not measurement; in agentic mode how
        often the agent searched is a result.
        """
        self._agentic = config.mode == "agentic"

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

    # -- lifecycle ---------------------------------------------------------

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

    def on_write(self, sample: Sample, session: Session, seconds: float) -> None:
        """Accumulate the sample's agent-driven write time."""
        state = self._state(sample)
        state["write_s"] = state.get("write_s", 0.0) + seconds

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """Book ingestion's time; a run that ingested nothing charges zero."""
        stats["ingest_s"] = self._elapsed(sample, "ingest") or 0.0
        state = self._state(sample)
        if "write_s" in state:
            stats["write_s"] = state["write_s"]

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
        if self._agentic:
            row["num_searches"] = state.get("num_searches", 0)
        answer_s = state.pop("answer_s", None)
        if answer_s is not None:
            row["answer_s"] = answer_s
