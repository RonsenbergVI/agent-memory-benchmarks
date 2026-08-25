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

import contextvars
from collections.abc import Callable
from typing import TYPE_CHECKING

from amb.base.callback import Callback
from amb.base.memory import Memory
from amb.constants import TOKEN_TRACKING_KEYS
from amb.contracts import QAPair, Run, Sample

if TYPE_CHECKING:
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

    Counters live in a `ContextVar`, not a `threading.local()`. Both keep
    one sample's spend off another's under `--workers` — a thread starts
    with its own empty context — but only the ContextVar survives the
    thread hop an integration makes when it marshals its SDK calls onto a
    background event loop. `asyncio.run_coroutine_threadsafe` copies the
    *submitting* thread's context, so the counters the worker installed
    are the ones the loop thread books into. Under thread-local storage
    those calls were intercepted, handed to `_record`, and silently
    dropped, and the system reported a confident zero it had not earned.
    """

    KEYS = TOKEN_TRACKING_KEYS

    def __init__(self) -> None:
        """Start with nothing patched and no sample being measured."""
        self._originals: list = []
        # the dict is shared by reference, so a call booked from another
        # thread that inherited this context lands in the sample's own
        # counters rather than a copy of them
        self._counters: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
            "amb_usage_counters", default=None
        )
        self._sample_mark: contextvars.ContextVar[dict] = contextvars.ContextVar(
            "amb_usage_sample_mark", default={}
        )
        self._question_mark: contextvars.ContextVar[dict] = contextvars.ContextVar(
            "amb_usage_question_mark", default={}
        )

    @property
    def counters(self) -> dict:
        """This context's counters; zeros when no sample is being measured."""
        counters = self._counters.get()
        return counters if counters is not None else dict.fromkeys(self.KEYS, 0)

    def on_run_begin(self, config: "RunConfig", run: Run) -> None:
        """Patch the SDK once for the whole run.

        Per-sample install/remove breaks under `--workers`: the first
        sample to finish would unpatch the SDK while others still run.
        """
        self._install()

    def on_run_end(self, run: Run) -> None:
        """Restore the SDK's original methods."""
        self._remove()

    def on_sample_begin(self, sample: Sample, system: Memory) -> None:
        """Start this context's counters for the sample."""
        self._counters.set(dict.fromkeys(self.KEYS, 0))
        self._sample_mark.set(self._snapshot())

    def on_ingest_end(self, sample: Sample, system: Memory, stats: dict) -> None:
        """Record ingestion's token spend on the sample's stats."""
        stats["token_usage"] = self._delta(self._sample_mark.get())

    def on_question_begin(self, sample: Sample, qa: QAPair) -> None:
        """Mark the counters so the question's own spend can be measured."""
        self._question_mark.set(self._snapshot())

    def on_question_end(self, sample: Sample, qa: QAPair, row: dict) -> None:
        """Record the question's search-time token spend on its row."""
        delta = self._delta(self._question_mark.get())
        row["memory_tokens"] = (
            delta["llm_input_tokens"]
            + delta["llm_output_tokens"]
            + delta["embedding_tokens"]
        )

    def on_sample_end(self, sample: Sample, system: Memory) -> None:
        """Stop measuring in this context."""
        self._counters.set(None)

    def _snapshot(self) -> dict:
        return dict(self.counters)

    def _delta(self, mark: dict) -> dict:
        return {key: self.counters[key] - mark.get(key, 0) for key in self.KEYS}

    def _record(self, kind: str, response: object) -> None:
        counters = self._counters.get()
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

    def _targets(self) -> list[tuple[type, str, str, bool]]:
        """Return every openai entry point that reports usage.

        Yields (owner, method, kind, is_async) quadruples. Whether a
        target is async is declared here rather than detected: the SDK's
        async methods are decorated, and `inspect.iscoroutinefunction`
        answers False for `AsyncCompletions.create` on openai 3.2/3.3
        even though awaiting it is the only way to call it. Detection
        therefore handed async chat completions to the *sync* wrapper,
        which "recorded" the un-awaited coroutine: the call was counted
        and every one of its tokens was lost. The resource classes are
        the reliable signal — Async* is async, and that cannot drift with
        a decorator change. openai is imported lazily so retrieval-only
        runs never touch it.
        """
        from openai.resources import embeddings
        from openai.resources.chat import completions

        targets: list[tuple[type, str, str, bool]] = [
            (completions.Completions, "create", "llm", False),
            (completions.Completions, "parse", "llm", False),
            (completions.AsyncCompletions, "create", "llm", True),
            (completions.AsyncCompletions, "parse", "llm", True),
            (embeddings.Embeddings, "create", "embedding", False),
            (embeddings.AsyncEmbeddings, "create", "embedding", True),
        ]
        try:
            from openai.resources import responses

            targets += [
                (responses.Responses, "create", "llm", False),
                (responses.AsyncResponses, "create", "llm", True),
                # graphiti-core calls parse(), which posts directly rather
                # than delegating to create()
                (responses.Responses, "parse", "llm", False),
                (responses.AsyncResponses, "parse", "llm", True),
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
        for owner, name, kind, is_async in targets:
            original = getattr(owner, name, None)
            if original is None:
                continue
            self._originals.append((owner, name, original))
            setattr(owner, name, self._wrap(original, kind, is_async))

    def _remove(self) -> None:
        for owner, name, original in self._originals:
            setattr(owner, name, original)
        self._originals.clear()

    def _wrap(self, original: Callable, kind: str, is_async: bool) -> Callable:
        """Wrap an SDK method so each response updates the counters."""
        if is_async:

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
