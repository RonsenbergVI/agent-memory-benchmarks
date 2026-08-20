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

"""The measurement callbacks: phase timing and the two token-usage trackers."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from amb.callbacks.openai import OpenAIUsageTracker
from amb.callbacks.tiktoken import TiktokenUsageTracker
from amb.callbacks.time import TimingTracker
from amb.constants import TOKEN_TRACKING_KEYS, RunType
from amb.contracts import (
    Conversation,
    IngestionRecord,
    MemoryHit,
    QAPair,
    Run,
    Sample,
    Session,
)
from amb.runner import RunConfig


def _sample(sample_id: str = "sample-1") -> Sample:
    return Sample(
        sample_id=sample_id,
        dataset="locomo",
        conversation=Conversation(conversation_id=f"conv-{sample_id}"),
    )


def _qa() -> QAPair:
    return QAPair(question_id="q1", question="Where does the dog sleep?")


def _run() -> Run:
    return Run(run_id="r1", system="fake", dataset="locomo")


def _zeros() -> dict:
    return dict.fromkeys(TOKEN_TRACKING_KEYS, 0)


# -- TimingTracker ----------------------------------------------------------


def test_ingest_end_books_elapsed_seconds(perf_clock, fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    tracker.on_ingest_begin(sample, memory)
    perf_clock["now"] = 1.5
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["ingest_s"] == 1.5


def test_ingest_end_without_begin_charges_zero(fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["ingest_s"] == 0.0
    assert "write_s" not in stats


def test_agent_writes_accumulate_into_write_s(fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    memory = fake_memory_class()
    session = Session(session_id="s1")
    tracker.on_sample_begin(sample, memory)
    tracker.on_ingest_begin(sample, memory)
    tracker.on_write(sample, session, 0.5)
    tracker.on_write(sample, session, 0.25)
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["write_s"] == 0.75


def test_direct_question_row_books_search_time_only(fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    qa = _qa()
    tracker.on_question_begin(sample, qa)
    tracker.on_search(sample, qa, [], 0.25)
    tracker.on_search(sample, qa, [], 0.5)
    row: dict = {}
    tracker.on_question_end(sample, qa, row)
    assert row["search_s"] == 0.75
    assert "num_searches" not in row
    assert "answer_s" not in row


def test_agentic_run_counts_searches_per_question(fake_memory_class):
    tracker = TimingTracker()
    tracker.on_run_begin(RunConfig(dataset="locomo", mode=RunType.AGENTIC), _run())
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    qa = _qa()
    tracker.on_question_begin(sample, qa)
    hits = [MemoryHit(content="a memory")]
    tracker.on_search(sample, qa, hits, 0.125)
    tracker.on_search(sample, qa, hits, 0.125)
    row: dict = {}
    tracker.on_question_end(sample, qa, row)
    assert row["search_s"] == 0.25
    assert row["num_searches"] == 2


def test_question_begin_resets_the_accumulators(fake_memory_class):
    tracker = TimingTracker()
    tracker.on_run_begin(RunConfig(dataset="locomo", mode=RunType.AGENTIC), _run())
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    qa = _qa()
    tracker.on_question_begin(sample, qa)
    tracker.on_search(sample, qa, [], 0.125)
    tracker.on_question_end(sample, qa, {})
    tracker.on_question_begin(sample, qa)
    row: dict = {}
    tracker.on_question_end(sample, qa, row)
    assert row["search_s"] == 0.0
    assert row["num_searches"] == 0


def test_answer_clock_books_answer_s_on_the_row(perf_clock, fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    qa = _qa()
    tracker.on_question_begin(sample, qa)
    perf_clock["now"] = 2.0
    tracker.on_answer_begin(sample, qa)
    perf_clock["now"] = 3.25
    tracker.on_answer_end(sample, qa)
    row: dict = {}
    tracker.on_question_end(sample, qa, row)
    assert row["answer_s"] == 1.25


def test_conversation_end_books_conversation_s(perf_clock, fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    tracker.on_conversation_begin(sample)
    perf_clock["now"] = 4.0
    record = IngestionRecord(sample_id=sample.sample_id, ingest_s=0.0, num_turns=0)
    tracker.on_conversation_end(sample, record)
    assert record.conversation_s == 4.0


def test_conversation_s_stays_none_when_never_begun(fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    tracker.on_sample_begin(sample, fake_memory_class())
    record = IngestionRecord(sample_id=sample.sample_id, ingest_s=0.0, num_turns=0)
    tracker.on_conversation_end(sample, record)
    assert record.conversation_s is None


def test_state_is_keyed_by_sample_for_concurrent_samples(perf_clock, fake_memory_class):
    tracker = TimingTracker()
    memory = fake_memory_class()
    first, second = _sample("a"), _sample("b")
    tracker.on_sample_begin(first, memory)
    tracker.on_sample_begin(second, memory)
    tracker.on_ingest_begin(first, memory)
    perf_clock["now"] = 1.0
    tracker.on_ingest_begin(second, memory)
    perf_clock["now"] = 3.0
    first_stats: dict = {}
    second_stats: dict = {}
    tracker.on_ingest_end(first, memory, first_stats)
    tracker.on_ingest_end(second, memory, second_stats)
    assert first_stats["ingest_s"] == 3.0
    assert second_stats["ingest_s"] == 2.0


def test_sample_end_drops_the_samples_clocks(perf_clock, fake_memory_class):
    tracker = TimingTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    tracker.on_ingest_begin(sample, memory)
    perf_clock["now"] = 5.0
    tracker.on_sample_end(sample, memory)
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["ingest_s"] == 0.0


# -- TiktokenUsageTracker ---------------------------------------------------


def test_ingest_token_usage_is_the_counter_delta(fake_memory_class):
    tracker = TiktokenUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    memory.counters = {
        "llm_input_tokens": 100,
        "llm_output_tokens": 40,
        "llm_calls": 2,
        "embedding_tokens": 900,
        "embedding_calls": 3,
    }
    tracker.on_sample_begin(sample, memory)
    memory.counters.update(
        llm_input_tokens=110,
        llm_output_tokens=45,
        llm_calls=3,
        embedding_tokens=950,
        embedding_calls=5,
    )
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["token_usage"] == {
        "llm_input_tokens": 10,
        "llm_output_tokens": 5,
        "llm_calls": 1,
        "embedding_tokens": 50,
        "embedding_calls": 2,
    }


def test_question_row_sums_token_deltas_but_not_call_counts(fake_memory_class):
    tracker = TiktokenUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    memory.counters = {"llm_input_tokens": 50, "llm_calls": 1}
    tracker.on_sample_begin(sample, memory)
    tracker.on_question_begin(sample, _qa())
    memory.counters.update(
        llm_input_tokens=57,
        llm_output_tokens=2,
        llm_calls=4,
        embedding_tokens=11,
        embedding_calls=9,
    )
    row: dict = {}
    tracker.on_question_end(sample, _qa(), row)
    assert row["memory_tokens"] == 20  # 7 + 2 + 11; the call counts don't sum


def test_missing_counter_keys_read_as_zero(fake_memory_class):
    tracker = TiktokenUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    memory.counters["embedding_tokens"] = 5
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["token_usage"] == {
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_calls": 0,
        "embedding_tokens": 5,
        "embedding_calls": 0,
    }


def test_sample_end_stops_reading_the_system(fake_memory_class):
    tracker = TiktokenUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    memory.counters = {"llm_input_tokens": 5}
    tracker.on_sample_begin(sample, memory)
    tracker.on_sample_end(sample, memory)
    memory.counters["llm_input_tokens"] = 500
    tracker.on_question_begin(sample, _qa())
    row: dict = {}
    tracker.on_question_end(sample, _qa(), row)
    assert row["memory_tokens"] == 0


def test_tiktoken_marks_are_thread_local_per_worker(fake_memory_class):
    tracker = TiktokenUsageTracker()
    main_memory, worker_memory = fake_memory_class(), fake_memory_class()
    tracker.on_sample_begin(_sample("main"), main_memory)
    main_memory.counters["llm_input_tokens"] = 7

    worker_usage: dict = {}

    def run_worker_sample() -> None:
        tracker.on_sample_begin(_sample("worker"), worker_memory)
        worker_memory.counters["llm_input_tokens"] = 3
        stats: dict = {}
        tracker.on_ingest_end(_sample("worker"), worker_memory, stats)
        worker_usage.update(stats["token_usage"])

    worker = threading.Thread(target=run_worker_sample)
    worker.start()
    worker.join()

    stats: dict = {}
    tracker.on_ingest_end(_sample("main"), main_memory, stats)
    assert worker_usage["llm_input_tokens"] == 3
    assert stats["token_usage"]["llm_input_tokens"] == 7


# -- OpenAIUsageTracker -----------------------------------------------------


def _chat_response(prompt: int, completion: int) -> SimpleNamespace:
    """A chat-completions-shaped response: prompt/completion token names."""
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    )


def test_counters_read_as_zero_outside_a_sample():
    assert OpenAIUsageTracker().counters == _zeros()


def test_chat_usage_books_prompt_and_completion_tokens(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    tracker._record("llm", _chat_response(12, 5))
    assert tracker.counters == {
        "llm_input_tokens": 12,
        "llm_output_tokens": 5,
        "llm_calls": 1,
        "embedding_tokens": 0,
        "embedding_calls": 0,
    }


def test_responses_usage_falls_back_to_input_output_tokens(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=8, output_tokens=2))
    tracker._record("llm", response)
    assert tracker.counters["llm_input_tokens"] == 8
    assert tracker.counters["llm_output_tokens"] == 2


def test_embedding_usage_books_prompt_tokens_or_total_tokens(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    tracker._record(
        "embedding",
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=13, total_tokens=99)),
    )
    tracker._record("embedding", SimpleNamespace(usage=SimpleNamespace(total_tokens=6)))
    assert tracker.counters["embedding_tokens"] == 19  # 13 (not 99) + 6
    assert tracker.counters["embedding_calls"] == 2
    assert tracker.counters["llm_calls"] == 0


def test_response_without_usage_still_counts_the_call(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    tracker._record("llm", SimpleNamespace())
    assert tracker.counters["llm_calls"] == 1
    assert tracker.counters["llm_input_tokens"] == 0


def test_record_outside_any_sample_is_dropped():
    tracker = OpenAIUsageTracker()
    tracker._record("llm", _chat_response(12, 5))
    assert tracker.counters == _zeros()


def test_question_delta_counts_only_question_phase_spend(fake_memory_class):
    tracker = OpenAIUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    tracker._record("llm", _chat_response(100, 10))
    stats: dict = {}
    tracker.on_ingest_end(sample, memory, stats)
    assert stats["token_usage"] == {
        "llm_input_tokens": 100,
        "llm_output_tokens": 10,
        "llm_calls": 1,
        "embedding_tokens": 0,
        "embedding_calls": 0,
    }
    qa = _qa()
    tracker.on_question_begin(sample, qa)
    tracker._record("llm", _chat_response(7, 2))
    tracker._record(
        "embedding",
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, total_tokens=11)),
    )
    row: dict = {}
    tracker.on_question_end(sample, qa, row)
    assert row["memory_tokens"] == 20  # 7 + 2 + 11; ingestion's spend excluded


def test_openai_counters_are_thread_local(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    tracker._record("llm", _chat_response(3, 1))

    seen: dict = {}

    def worker() -> None:
        seen["counters"] = dict(tracker.counters)
        # no sample began on this thread, so this spend has nowhere to book
        tracker._record("llm", _chat_response(500, 500))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["counters"] == _zeros()
    assert tracker.counters["llm_input_tokens"] == 3
    assert tracker.counters["llm_calls"] == 1


def test_sample_end_resets_this_threads_counters(fake_memory_class):
    tracker = OpenAIUsageTracker()
    sample = _sample()
    memory = fake_memory_class()
    tracker.on_sample_begin(sample, memory)
    tracker._record("llm", _chat_response(3, 1))
    tracker.on_sample_end(sample, memory)
    assert tracker.counters == _zeros()


def test_sync_wrapper_returns_the_response_and_records(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    response = _chat_response(4, 6)
    calls: list = []

    def original(client_self, *args, **kwargs):
        calls.append((client_self, args, kwargs))
        return response

    wrapped = tracker._wrap(original, "llm")
    client = object()
    result = wrapped(client, "model", stream=False)
    assert result is response
    assert calls == [(client, ("model",), {"stream": False})]
    assert tracker.counters["llm_input_tokens"] == 4


def test_async_wrapper_awaits_and_records(fake_memory_class):
    tracker = OpenAIUsageTracker()
    tracker.on_sample_begin(_sample(), fake_memory_class())
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=8, output_tokens=2))

    async def original(client_self, *args, **kwargs):
        return response

    wrapped = tracker._wrap(original, "llm")
    result = asyncio.run(wrapped(object()))
    assert result is response
    assert tracker.counters["llm_output_tokens"] == 2
    assert tracker.counters["llm_calls"] == 1


def test_run_begin_patches_the_sdk_and_run_end_restores_it():
    pytest.importorskip("openai")
    from openai.resources.chat import completions

    tracker = OpenAIUsageTracker()
    config = RunConfig(dataset="locomo")
    original = completions.Completions.create
    tracker.on_run_begin(config, _run())
    try:
        assert completions.Completions.create is not original
        # a second install is a no-op, so one remove still restores
        tracker.on_run_begin(config, _run())
    finally:
        tracker.on_run_end(_run())
    assert completions.Completions.create is original
