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

import inspect
import threading
from collections.abc import Callable

from amb.base import Callback, Memory
from amb.constants import TOKEN_TRACKING_KEYS
from amb.contracts import QAPair, Run, Sample
from amb.runner import RunConfig


class OpenAIUsageTracker(Callback):
    """Counts tokens for memory systems that call the openai SDK in-process.

    Attached by default on every Benchmark: wrappers on the SDK's sync and
    async request methods read the billed ``usage`` off every response
    (mem0, graphiti, fraise), so the report gets a token_usage dict per
    sample (ingestion) and a memory_tokens delta per question (search); a
    system that makes no in-process calls records a truthful zero. Systems
    that spend inside their own server attach their own counting strategy
    instead (letta's TiktokenUsageTracker). Class-level patching is
    process-wide — fine here, the benchmark runs one system instance at a
    time.
    """

    KEYS = TOKEN_TRACKING_KEYS

    def __init__(self) -> None:
        """Start with nothing patched and no sample being measured.

        Counters are thread-local: with `--workers`, each sample runs its
        whole lifecycle on one thread, so per-sample and per-question
        deltas stay correctly attributed under concurrency.
        """
        self._originals: list = []
        self._local = threading.local()

    @property
    def counters(self) -> dict:
        """This thread's counters; zeros when no sample is being measured."""
        counters = getattr(self._local, "counters", None)
        return counters if counters is not None else dict.fromkeys(self.KEYS, 0)

    def on_run_begin(self, config: RunConfig, run: Run) -> None:
        """Patch the SDK once for the whole run.

        Per-sample install/remove breaks under `--workers`: the first
        sample to finish would unpatch the SDK while others still run.
        """
        self._install()

    def on_run_end(self, run: Run) -> None:
        """Restore the SDK's original methods."""
        self._remove()

    def on_sample_begin(self, sample: Sample, system: Memory) -> None:
        """Start this thread's counters for the sample."""
        self._local.counters = dict.fromkeys(self.KEYS, 0)
        self._local.sample_mark = self._snapshot()

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """Record ingestion's token spend on the sample's stats."""
        stats["token_usage"] = self._delta(self._local.sample_mark)

    def on_question_begin(self, sample: Sample, qa: QAPair) -> None:
        """Mark the counters so the question's own spend can be measured."""
        self._local.question_mark = self._snapshot()

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
        self._local.counters = None

    def _snapshot(self) -> dict:
        return dict(self.counters)

    def _delta(self, mark: dict) -> dict:
        return {key: self.counters[key] - mark.get(key, 0) for key in self.KEYS}

    def _record(self, kind: str, response: object) -> None:
        counters = getattr(self._local, "counters", None)
        if counters is None:
            return  # a call outside any sample's lifecycle
        usage = getattr(response, "usage", None)
        if kind == "embedding":
            counters["embedding_calls"] += 1
            if usage is not None:
                counters["embedding_tokens"] += (
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "total_tokens", 0)
                    or 0
                )
        else:
            counters["llm_calls"] += 1
            if usage is not None:
                # chat completions say prompt/completion, responses API
                # says input/output
                counters["llm_input_tokens"] += (
                    getattr(usage, "prompt_tokens", None)
                    or getattr(usage, "input_tokens", 0)
                    or 0
                )
                counters["llm_output_tokens"] += (
                    getattr(usage, "completion_tokens", None)
                    or getattr(usage, "output_tokens", 0)
                    or 0
                )

    def _targets(self) -> list[tuple[type, str, str]]:
        """Return every openai entry point that reports usage.

        Yields (owner, method, kind) triples. openai is imported lazily so
        retrieval-only runs never touch it.
        """
        from openai.resources import embeddings
        from openai.resources.chat import completions

        targets: list[tuple[type, str, str]] = [
            (completions.Completions, "create", "llm"),
            (completions.Completions, "parse", "llm"),
            (completions.AsyncCompletions, "create", "llm"),
            (completions.AsyncCompletions, "parse", "llm"),
            (embeddings.Embeddings, "create", "embedding"),
            (embeddings.AsyncEmbeddings, "create", "embedding"),
        ]
        try:
            from openai.resources import responses

            targets += [
                (responses.Responses, "create", "llm"),
                (responses.AsyncResponses, "create", "llm"),
                # graphiti-core calls parse(), which posts directly rather
                # than delegating to create()
                (responses.Responses, "parse", "llm"),
                (responses.AsyncResponses, "parse", "llm"),
            ]
        except ImportError:
            pass
        return targets

    def _install(self) -> None:
        if self._originals:
            return
        try:
            targets = self._targets()
        except ImportError:
            # no openai SDK in this env: nothing in-process to count (the
            # proxy feed, when wired, still works)
            return
        for owner, name, kind in targets:
            original = getattr(owner, name, None)
            if original is None:
                continue
            self._originals.append((owner, name, original))
            setattr(owner, name, self._wrap(original, kind))

    def _remove(self) -> None:
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()

    def _wrap(self, original: Callable, kind: str) -> Callable:
        """Wrap an SDK method so each response updates the counters."""
        if inspect.iscoroutinefunction(original):

            async def wrapper(
                client_self: object, *args: object, **kwargs: object
            ) -> object:
                response = await original(client_self, *args, **kwargs)
                self._record(kind, response)
                return response

        else:

            def wrapper(client_self: object, *args: object, **kwargs: object) -> object:
                response = original(client_self, *args, **kwargs)
                self._record(kind, response)
                return response

        return wrapper
