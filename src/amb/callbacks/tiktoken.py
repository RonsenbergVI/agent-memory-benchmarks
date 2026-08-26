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

from amb.base.callback import Callback
from amb.base.memory import Memory
from amb.constants import TOKEN_TRACKING_KEYS
from amb.contracts import QAPair, Sample


class TiktokenUsageTracker(Callback):
    """Books token spend the memory system computes about itself.

    For systems that spend inside their own server, invisible to the SDK
    wrappers: the adapter reports its computed spend through
    `Memory.usage_counters()` (letta: tiktoken over every stored passage
    and query — exact for embeddings, which bill their input verbatim),
    and this callback writes the same report fields as OpenAIUsageTracker.
    Opt-in via `callback_classes` (letta today). Marks are thread-local
    and each sample has its own system instance, so attribution holds
    under `--workers`.
    """

    KEYS = TOKEN_TRACKING_KEYS

    def __init__(self) -> None:
        """Start with no sample being measured."""
        self._local = threading.local()

    def _counters(self) -> dict:
        system = getattr(self._local, "system", None)
        return system.usage_counters() if system is not None else {}

    def _delta(self, mark: dict) -> dict:
        counters = self._counters()
        return {key: counters.get(key, 0) - mark.get(key, 0) for key in self.KEYS}

    def on_sample_begin(self, sample: Sample, system: Memory) -> None:
        """Bind this thread to its sample's system and mark its counters."""
        self._local.system = system
        self._local.sample_mark = system.usage_counters()

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """Record ingestion's token spend on the sample's stats."""
        stats["token_usage"] = self._delta(self._local.sample_mark)

    def on_question_begin(self, sample: Sample, qa: QAPair) -> None:
        """Mark the counters so the question's own spend can be measured."""
        self._local.question_mark = self._counters()

    def on_question_end(self, sample: Sample, qa: QAPair, row: dict) -> None:
        """Record the question's search-time token spend on its row."""
        delta = self._delta(self._local.question_mark)
        row["memory_tokens"] = (
            delta["llm_input_tokens"]
            + delta["llm_output_tokens"]
            + delta["embedding_tokens"]
        )

    def on_sample_end(self, sample: Sample, system: Memory) -> None:
        """Stop measuring on this thread."""
        self._local.system = None
