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

from typing import Any, ClassVar

from amb.base import Metric
from amb.base.metric import Record


class Count(Metric):
    """How many observations carried the key at all (e.g. errors)."""

    report_when_empty = True

    def accumulate(self, value: Any) -> None:
        """Counting happens in update_state, so nothing to accumulate."""

    def result(self) -> int:
        """Return how many records carried the field."""
        return self.count


class DictSum(Metric):
    """Key-wise sums over dict-valued observations.

    For fields whose keys the data producer declares, e.g. ``token_usage``.
    """

    def __init__(self, name: str, key: str | None = None) -> None:
        """Start with no keys; whatever the data carries defines them."""
        super().__init__(name, key)
        self._totals: dict = {}

    def accumulate(self, value: dict) -> None:
        """Add each key of the observed dict into the running totals."""
        for k, v in value.items():
            self._totals[k] = self._totals.get(k, 0) + v

    def result(self) -> dict:
        """Return the per-key totals, key-sorted for stable reports."""
        return dict(sorted(self._totals.items()))

    def reset_state(self) -> None:
        """Clear the accumulated per-key totals."""
        super().reset_state()
        self._totals = {}


class Sum(Metric):
    """Running total of the observed values."""

    def __init__(self, name: str, key: str | None = None) -> None:
        """Start the total at zero."""
        super().__init__(name, key)
        self._total: float = 0

    def accumulate(self, value: Any) -> None:
        """Add the observed value to the total."""
        self._total += value

    def result(self) -> float:
        """Return the total."""
        return self._total

    def reset_state(self) -> None:
        """Clear the total."""
        super().reset_state()
        self._total = 0


class Mean(Metric):
    """Arithmetic mean of the observed values; 0.0 when nothing was seen."""

    def __init__(self, name: str, key: str | None = None) -> None:
        """Start the running total at zero."""
        super().__init__(name, key)
        self._total = 0.0

    def accumulate(self, value: Any) -> None:
        """Add the observed value to the running total."""
        self._total += value

    def result(self) -> float:
        """Return the mean over the records that carried the field."""
        return self._total / self.count if self.count else 0.0

    def reset_state(self) -> None:
        """Clear the running total."""
        super().reset_state()
        self._total = 0.0


class LatencyPercentiles(Metric):
    """p50/p95/p99 over observed latencies, in seconds."""

    def __init__(
        self, name: str, key: str | None = None, points: tuple[int, ...] = (50, 95, 99)
    ) -> None:
        """Record which percentiles to report (default p50/p95/p99)."""
        super().__init__(name, key)
        self.points = points
        self._values: list[float] = []

    def accumulate(self, value: Any) -> None:
        """Keep the observed latency; percentiles need the whole sample."""
        self._values.append(value)

    @staticmethod
    def percentile(values: list[float], pct: float) -> float:
        """Nearest-rank percentile; 0.0 for an empty list."""
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
        return ordered[idx]

    def result(self) -> dict[str, float]:
        """Return each configured percentile, keyed ``p<n>_s``."""
        return {f"p{p}_s": self.percentile(self._values, p) for p in self.points}

    def reset_state(self) -> None:
        """Discard the collected latencies."""
        super().reset_state()
        self._values = []


class Attribution(Metric):
    """Who lost a question: the memory that missed, or the agent that fumbled.

    Each judged question with evidence labels lands in one of four cells,
    crossing "every evidence session retrieved" with the judge's verdict:
    `correct`, `agent_fault` (found, wrong), `memory_fault` (missed, wrong)
    and `correct_despite_miss` (missed, right). Two rates factor them: `found` is the
    memory's, `correct_when_found` the agent's — flat across systems unless
    a memory hands back evidence in a form the agent cannot use. Questions
    without evidence labels or a verdict are left out, never scored as misses.
    """

    CELLS: ClassVar[tuple[str, ...]] = (
        "correct",
        "agent_fault",
        "memory_fault",
        "correct_despite_miss",
    )

    def __init__(self, name: str = "attribution") -> None:
        """Start with every cell empty."""
        super().__init__(name)
        self._cells: dict[str, int] = dict.fromkeys(self.CELLS, 0)

    def update_state(self, record: Record) -> None:
        """Place the record in its cell when it carries labels and a verdict."""
        retrieved = getattr(record, "retrieved_session_ids", None)
        evidence = getattr(record, "evidence_session_ids", None)
        verdict = getattr(record, "judge_correct", None)
        if retrieved is None or not evidence or verdict is None:
            return
        self.accumulate((set(evidence) <= set(retrieved), bool(verdict)))
        self.count += 1

    def accumulate(self, value: tuple[bool, bool]) -> None:
        """Count one (found, correct) pair in its cell."""
        found, correct = value
        if found:
            self._cells["correct" if correct else "agent_fault"] += 1
        else:
            self._cells["correct_despite_miss" if correct else "memory_fault"] += 1

    def result(self) -> dict[str, float | int]:
        """Return each cell's share, each branch's rate and its split, and `n`."""
        n = self.count
        c = self._cells
        found = c["correct"] + c["agent_fault"]
        missed = n - found
        shares = {cell: (c[cell] / n if n else 0.0) for cell in self.CELLS}
        correct_when_found = c["correct"] / found if found else 0.0
        correct_when_missed = c["correct_despite_miss"] / missed if missed else 0.0
        return {
            **shares,
            "found": found / n if n else 0.0,
            "missed": missed / n if n else 0.0,
            "correct_when_found": correct_when_found,
            "correct_when_missed": correct_when_missed,
            # accuracy the memory added over the agent's rate without evidence
            "lift": (found / n) * (correct_when_found - correct_when_missed)
            if n
            else 0.0,
            "n": n,
        }

    def reset_state(self) -> None:
        """Empty every cell."""
        super().reset_state()
        self._cells = dict.fromkeys(self.CELLS, 0)


class MemoryLift(Metric):
    """The memory's contribution as one top-level number: `Attribution`'s lift.

    `found × (correct_when_found − correct_when_missed)`: the accuracy points
    the memory added over what the agent answers without the evidence. A
    memory that finds everything but returns it unusably earns little; so
    does one that returns clean facts but misses half the questions.
    """

    def __init__(self, name: str = "memory_lift") -> None:
        """Report under `memory_lift`, a top-level float the tables pick up."""
        super().__init__(name)
        self._cells = Attribution()

    def update_state(self, record: Record) -> None:
        """Place the record as `Attribution` does."""
        self._cells.update_state(record)
        self.count = self._cells.count

    def accumulate(self, value: Any) -> None:
        """Accumulation happens in update_state, so nothing to do."""

    def result(self) -> float:
        """Return the lift."""
        return float(self._cells.result()["lift"])

    def reset_state(self) -> None:
        """Empty the cells."""
        super().reset_state()
        self._cells.reset_state()


def default_metrics() -> list[Metric]:
    """The standard metric set a run is scored with when none is given.

    Dotted names land as nested summary sections: `memory_tokens` feeds the
    summary's headline cost total, `search_latency` the comparison's p50 column.
    """
    return [
        Sum("ingest.total_s", "ingest_s"),
        Sum("ingest.total_turns", "num_turns"),
        Sum("ingest.total_sessions", "num_sessions"),
        Sum("ingest.questions_dropped", "questions_dropped"),
        Sum("conversation.total_s", "conversation_s"),
        # k is the budget granted; this is what the system spent of it
        Mean("hits_per_query", "num_hits"),
        Mean("judge_accuracy", "judge_correct"),
        Attribution(),
        MemoryLift(),
        LatencyPercentiles("search_latency", "search_s"),
        LatencyPercentiles("answer_latency", "answer_s"),
        Mean("tokens.input_per_question", "input_tokens"),
        Mean("tokens.output_per_question", "output_tokens"),
        DictSum("memory_tokens.ingest", "token_usage"),
        Sum("memory_tokens.search_total", "memory_tokens"),
        Count("errors", "error"),
    ]


def default_category_metrics() -> list[Metric]:
    """Fresh instances of the per-question-category metric set.

    A factory because each category accumulates its own state; row-level
    metrics only, since categories are a property of questions, not ingestion.
    """
    return [
        Mean("judge_accuracy", "judge_correct"),
        Attribution(),
        MemoryLift(),
    ]
