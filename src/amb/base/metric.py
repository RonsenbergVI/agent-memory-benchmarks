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

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from amb.contracts.run import IngestionRecord, QuestionRecord

Record = QuestionRecord | IngestionRecord


class Metric(ABC):
    """A running metric over observation records.

    update_state() takes a whole record, grabs the field named by ``key``
    (default: the metric's name), and accumulates it; records without the
    field are ignored. result() produces the final value, reset_state()
    starts a new scope.
    """

    # report with zero observations: an error count of 0 is a statement
    report_when_empty: ClassVar[bool] = False

    def __init__(self, name: str, key: str | None = None) -> None:
        """Name the metric and the record field it reads (default: its name)."""
        self.name = name
        self.key = key or name
        self.count = 0

    def update_state(self, record: Record) -> None:
        """Fold one observation record in, ignoring it if the field is absent."""
        value = getattr(record, self.key, None)
        if value is not None:
            self.accumulate(value)
            self.count += 1

    @abstractmethod
    def accumulate(self, value: Any) -> None:
        """Add one observed value to the running state."""

    @abstractmethod
    def result(self) -> object:
        """Compute the final value from the accumulated state."""

    def reset_state(self) -> None:
        """Clear the accumulated state, starting a new scope."""
        self.count = 0


class Scorer(Metric):
    """A metric that scores a (predicted, gold) pair per observation.

    ``predicted_key``/``gold_key`` name the fields holding the pair;
    accumulation comes from the mixed-in accumulator, e.g.
    ``class AnswerF1(Scorer, Mean)``. Records missing the prediction or
    with an empty gold value are ignored.
    """

    predicted_key: ClassVar[str] = "predicted_answer"
    gold_key: ClassVar[str] = "gold_answer"

    def update_state(self, record: Record) -> None:
        """Score the record's (predicted, gold) pair and accumulate it."""
        predicted = getattr(record, self.predicted_key, None)
        gold = getattr(record, self.gold_key, None)
        if predicted is None or not gold:
            return
        self.accumulate(self.score(predicted, gold))
        self.count += 1

    @abstractmethod
    def score(self, predicted: Any, gold: Any) -> float:
        """Score one prediction against its ground truth."""
