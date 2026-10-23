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

import math
from typing import Any, ClassVar

from amb.base import Metric


class Count(Metric):
    """How many observations carried the key at all (e.g. errors)."""

    report_when_empty = True

    def accumulate(self, value: Any) -> None:
        """Counting happens in update_state, so nothing to accumulate."""

    def result(self) -> int:
        """Return how many records carried the field."""
        return self.count


class ValueCounts(Metric):
    """How many records carried each distinct value of the field."""

    def __init__(self, name: str, key: str | None = None) -> None:
        """Start with no values seen."""
        super().__init__(name, key)
        self._counts: dict[str, int] = {}

    def accumulate(self, value: Any) -> None:
        """Count one occurrence of the observed value."""
        self._counts[str(value)] = self._counts.get(str(value), 0) + 1

    def result(self) -> dict[str, int]:
        """Return occurrences per distinct value, key-sorted."""
        return dict(sorted(self._counts.items()))

    def reset_state(self) -> None:
        """Forget every value seen."""
        super().reset_state()
        self._counts = {}


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


class WilsonInterval(Metric):
    """The 95% Wilson interval of a binary outcome's success rate.

    Accuracy over a few hundred verdicts is a few points of noise either
    way; the interval travels with it so two systems inside each other's
    band are never ranked as if they differed. Wilson rather than the
    normal approximation: it stays inside 0–1 and honest near the ends.
    """

    Z: ClassVar[float] = 1.96

    def __init__(self, name: str, key: str | None = None) -> None:
        """Start with no verdicts."""
        super().__init__(name, key)
        self._successes = 0

    def accumulate(self, value: Any) -> None:
        """Count one verdict."""
        self._successes += bool(value)

    def result(self) -> dict[str, float]:
        """Return the interval bounds and the number of verdicts behind them."""
        n, z = self.count, self.Z
        if not n:
            return {"low": 0.0, "high": 0.0, "n": 0}
        p = self._successes / n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return {"low": max(0.0, centre - half), "high": min(1.0, centre + half), "n": n}

    def reset_state(self) -> None:
        """Forget every verdict."""
        super().reset_state()
        self._successes = 0


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
        WilsonInterval("judge_accuracy_ci", "judge_correct"),
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
        WilsonInterval("judge_accuracy_ci", "judge_correct"),
    ]
