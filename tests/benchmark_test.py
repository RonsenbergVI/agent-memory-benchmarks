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

"""The Benchmark declaration: params, factory hooks, and the bare-Memory wrapper."""

import pytest

from amb.base.benchmark import Benchmark
from amb.base.callback import Callback, Callbacks
from amb.base.memory import Memory
from amb.callbacks import OpenAIUsageTracker
from amb.contracts import Conversation, Sample, Session


class FakeSearchToolset:
    def __init__(self, memory: Memory, conversation_id: str, k: int = 10) -> None:
        self.memory = memory
        self.conversation_id = conversation_id
        self.k = k


class FakeIngestToolset:
    def __init__(self, memory: Memory, conversation_id: str, session: Session) -> None:
        self.memory = memory
        self.conversation_id = conversation_id
        self.session = session


class TrackerA(Callback):
    pass


class TrackerB(Callback):
    pass


class FakeBenchmark(Benchmark):
    name = "fakebench"
    default_params = {"model": "m-default", "k": 5}


class SearchOnlyBenchmark(FakeBenchmark):
    search_toolset_class = FakeSearchToolset


class IngestOnlyBenchmark(FakeBenchmark):
    ingest_toolset_class = FakeIngestToolset


class AgenticBenchmark(FakeBenchmark):
    search_toolset_class = FakeSearchToolset
    ingest_toolset_class = FakeIngestToolset


class OrderedBenchmark(FakeBenchmark):
    callback_classes = (TrackerA, TrackerB)


def _sample() -> Sample:
    return Sample(
        sample_id="sample-1",
        dataset="unit",
        conversation=Conversation(conversation_id="conv-1"),
    )


def test_params_default_to_empty_dict() -> None:
    """No overrides passed means an empty — not None — params dict."""
    assert Benchmark().params == {}
    assert Benchmark(None).params == {}


def test_params_are_copied_not_aliased() -> None:
    """Later mutation of the caller's dict cannot change the run's identity."""
    overrides = {"k": 7}
    benchmark = Benchmark(overrides)
    overrides["k"] = 99
    assert benchmark.params == {"k": 7}


def test_params_hold_only_the_explicit_overrides() -> None:
    """default_params stays out of self.params; it merges at create_system."""
    benchmark = FakeBenchmark({"k": 7})
    assert benchmark.params == {"k": 7}


def test_declared_class_defaults() -> None:
    """The base declaration: unnamed, no defaults, no tool surface."""
    assert Benchmark.name == "unnamed"
    assert Benchmark.default_params == {}
    assert Benchmark.search_toolset_class is None
    assert Benchmark.ingest_toolset_class is None


def test_create_system_applies_overrides_on_top_of_defaults(fake_memory_class) -> None:
    """--param overrides win over default_params, key by key."""

    class SystemBenchmark(FakeBenchmark):
        system_class = fake_memory_class

    system = SystemBenchmark({"k": 7}).create_system()
    assert type(system) is fake_memory_class
    assert system.params == {"model": "m-default", "k": 7}


def test_create_system_uses_defaults_when_no_overrides(fake_memory_class) -> None:
    """With no --param, the system is built from default_params alone."""

    class SystemBenchmark(FakeBenchmark):
        system_class = fake_memory_class

    system = SystemBenchmark().create_system()
    assert system.params == {"model": "m-default", "k": 5}


def test_create_system_leaves_default_params_untouched(fake_memory_class) -> None:
    """Merging never mutates the class-level default_params dict."""

    class SystemBenchmark(FakeBenchmark):
        system_class = fake_memory_class

    SystemBenchmark({"model": "override"}).create_system()
    assert FakeBenchmark.default_params == {"model": "m-default", "k": 5}


def test_create_callbacks_instantiates_declared_classes_in_order() -> None:
    """One instance per declared class, in declaration order."""
    callbacks = OrderedBenchmark().create_callbacks()
    assert isinstance(callbacks, Callbacks)
    assert [type(callback) for callback in callbacks.callbacks] == [TrackerA, TrackerB]


def test_create_callbacks_returns_fresh_instances_per_call() -> None:
    """Each run gets its own callback instances, never shared state."""
    benchmark = OrderedBenchmark()
    first = benchmark.create_callbacks().callbacks
    second = benchmark.create_callbacks().callbacks
    assert first[0] is not second[0]
    assert first[1] is not second[1]


def test_every_benchmark_gets_the_usage_tracker_by_default() -> None:
    """The token tracker is the one default callback on every benchmark."""
    assert Benchmark.callback_classes == (OpenAIUsageTracker,)
    callbacks = Benchmark().create_callbacks().callbacks
    assert len(callbacks) == 1
    assert type(callbacks[0]) is OpenAIUsageTracker


@pytest.mark.parametrize(
    "benchmark_class, expected",
    [
        (FakeBenchmark, False),
        (SearchOnlyBenchmark, False),
        (IngestOnlyBenchmark, False),
        (AgenticBenchmark, True),
    ],
)
def test_supports_agentic_requires_both_toolset_halves(
    benchmark_class, expected
) -> None:
    """Agentic mode needs both the search and the ingest surface."""
    assert benchmark_class.supports_agentic() is expected


def test_create_search_toolset_binds_system_conversation_and_k(
    fake_memory_class,
) -> None:
    """The factory passes the system, conversation id, and k straight through."""

    class SystemAgenticBenchmark(AgenticBenchmark):
        system_class = fake_memory_class

    benchmark = SystemAgenticBenchmark()
    system = benchmark.create_system()
    toolset = benchmark.create_search_toolset(system, "conv-1", k=3)
    assert type(toolset) is FakeSearchToolset
    assert toolset.memory is system
    assert toolset.conversation_id == "conv-1"
    assert toolset.k == 3


def test_create_search_toolset_defaults_k_to_ten(fake_memory_class) -> None:
    """k falls back to 10 when the caller does not choose one."""

    class SystemAgenticBenchmark(AgenticBenchmark):
        system_class = fake_memory_class

    benchmark = SystemAgenticBenchmark()
    toolset = benchmark.create_search_toolset(benchmark.create_system(), "conv-1")
    assert toolset.k == 10


def test_create_ingest_toolset_binds_system_conversation_and_session(
    fake_memory_class,
) -> None:
    """The factory passes the system, conversation id, and session through."""

    class SystemAgenticBenchmark(AgenticBenchmark):
        system_class = fake_memory_class

    benchmark = SystemAgenticBenchmark()
    system = benchmark.create_system()
    session = Session(session_id="s1")
    toolset = benchmark.create_ingest_toolset(system, "conv-1", session)
    assert type(toolset) is FakeIngestToolset
    assert toolset.memory is system
    assert toolset.conversation_id == "conv-1"
    assert toolset.session is session


def test_toolset_factories_assert_when_the_surface_is_missing(
    fake_memory_class,
) -> None:
    """Calling a factory on a benchmark without that surface trips its assert."""

    class SystemBenchmark(FakeBenchmark):
        system_class = fake_memory_class

    benchmark = SystemBenchmark()
    system = benchmark.create_system()
    with pytest.raises(AssertionError):
        benchmark.create_search_toolset(system, "conv-1")
    with pytest.raises(AssertionError):
        benchmark.create_ingest_toolset(system, "conv-1", Session(session_id="s1"))


def test_for_system_wraps_a_bare_memory(fake_memory_class) -> None:
    """A bare Memory gets a named Benchmark subclass bound to it."""
    wrapper = Benchmark.for_system(fake_memory_class)
    assert issubclass(wrapper, Benchmark)
    assert wrapper.__name__ == "FakeMemoryBenchmark"
    assert wrapper.name == "fake"
    assert wrapper.system_class is fake_memory_class


def test_for_system_wrapper_builds_the_memory_with_no_params(
    fake_memory_class,
) -> None:
    """The wrapper's create_system builds the memory with empty params."""
    system = Benchmark.for_system(fake_memory_class)().create_system()
    assert type(system) is fake_memory_class
    assert system.params == {}


def test_for_system_wrapper_lacks_the_agentic_surface(fake_memory_class) -> None:
    """The wrapper sets only name and system_class — no toolsets, no extras."""
    wrapper = Benchmark.for_system(fake_memory_class)
    assert wrapper.search_toolset_class is None
    assert wrapper.ingest_toolset_class is None
    assert wrapper.supports_agentic() is False
    assert wrapper.default_params == {}
    assert wrapper.callback_classes == (OpenAIUsageTracker,)


def test_for_system_derives_from_the_class_it_is_called_on(fake_memory_class) -> None:
    """for_system builds on cls, so a subclass's hooks carry over."""

    class OtherMemory(fake_memory_class):
        name = "other"

    wrapper = AgenticBenchmark.for_system(OtherMemory)
    assert issubclass(wrapper, AgenticBenchmark)
    assert wrapper.name == "other"
    assert wrapper.system_class is OtherMemory
    assert wrapper.supports_agentic() is True


def test_sample_hooks_default_to_no_ops(fake_memory_class) -> None:
    """before_sample and after_sample do nothing unless overridden."""

    class SystemBenchmark(FakeBenchmark):
        system_class = fake_memory_class

    benchmark = SystemBenchmark()
    system = benchmark.create_system()
    sample = _sample()
    assert benchmark.before_sample(system, sample) is None
    assert benchmark.after_sample(system, sample) is None
