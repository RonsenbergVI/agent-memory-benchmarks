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

from abc import ABC

from amb.base import Memory
from amb.contracts import IngestionRecord, MemoryHit, QAPair, Run, Sample, Session
from amb.runner import RunConfig


class Callback(ABC):
    """Base callback: every hook is a no-op; override the ones needed."""

    def on_run_begin(self, config: RunConfig, run: Run) -> None:
        """Before the first sample is loaded."""

    def on_run_end(self, run: Run) -> None:
        """After the last sample, before the data is returned."""

    def on_sample_begin(self, sample: Sample, system: Memory) -> None:
        """After the memory system is set up, before ingestion."""

    def on_sample_end(self, sample: Sample, system: Memory) -> None:
        """After the sample's questions, before teardown."""

    def on_ingest_begin(self, sample: Sample, system: Memory) -> None:
        """Before the sample's first session is stored.

        Not fired at all when there is nothing to ingest (`--reuse`), so an
        observer can tell "no ingestion" from "instant ingestion".
        """

    def on_session_begin(
        self, sample: Sample, session: Session, system: Memory
    ) -> None:
        """Before one session is stored, in either mode."""

    def on_session_end(self, sample: Sample, session: Session, system: Memory) -> None:
        """After one session is stored, in either mode."""

    def on_write(self, sample: Sample, session: Session, seconds: float) -> None:
        """After one agent-driven write, with what it cost (agentic mode).

        Fired by IngestToolset.record_write, so every integration reports
        its writes through the one call it already makes.
        """

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """After a sample's ingestion; may enrich the ingest stats dict."""

    def on_conversation_begin(self, sample: Sample) -> None:
        """Before the sample's first question."""

    def on_conversation_end(self, sample: Sample, stats: IngestionRecord) -> None:
        """After the sample's last question; may enrich its stats record.

        The record is already validated by now, so this hook sets attributes
        on it rather than keys in a dict.
        """

    def on_question_begin(self, sample: Sample, qa: QAPair) -> None:
        """Before one question is asked."""

    def on_search_begin(self, sample: Sample, qa: QAPair) -> None:
        """Before a harness-driven search (direct mode only).

        An agentic run's searches are the agent's to schedule and happen
        inside the toolset, which reports each one through `on_search`.
        """

    def on_search(
        self,
        sample: Sample,
        qa: QAPair,
        hits: list[MemoryHit],
        seconds: float,
    ) -> None:
        """After one search, with what it returned and what it cost.

        The one search event both modes share: the Runner fires it for the
        search it drives, SearchToolset.record fires it for each search an
        agent drives. Whoever owns the boundary times it, since only they
        can see it.
        """

    def on_answer_begin(self, sample: Sample, qa: QAPair) -> None:
        """Before an answer is generated (direct with a model, or agentic)."""

    def on_answer_end(self, sample: Sample, qa: QAPair) -> None:
        """After an answer is generated."""

    def on_question_end(self, sample: Sample, qa: QAPair, row: dict) -> None:
        """After one question; may enrich the report row."""
