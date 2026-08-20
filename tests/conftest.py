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

"""Every pytest fixture of the unit suite, in one place.

The suite runs with --import-mode=importlib and no __init__.py files, so test
modules cannot import helpers from this file: anything shared is exposed as a
fixture instead, returning a class or value for tests to instantiate,
subclass, or copy.
"""

import copy
import io
import json
import sys
from pathlib import Path

import pytest

from amb import logs
from amb.base.memory import Memory
from amb.contracts import MemoryHit, Sample, Session
from amb.datasets.longmemeval import LongMemEvalLoader


class FakeMemory(Memory):
    """In-test memory: stores nothing, remembers nothing.

    Serves two suites at once: the Benchmark factory tests build it through
    create_system (so the constructor forwards arbitrary params), and the
    callback tests read back whatever counters a test sets on `counters`.

    The class name is load-bearing: Benchmark.for_system derives the wrapper
    name "FakeMemoryBenchmark" from it.
    """

    name = "fake"
    description = "in-test memory that stores nothing"

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.counters: dict = {}

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Store nothing."""
        return None

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Remember nothing."""
        return []

    def usage_counters(self) -> dict:
        """Report a snapshot of the test-controlled counters."""
        return dict(self.counters)


@pytest.fixture
def fake_memory_class() -> type[FakeMemory]:
    """The FakeMemory class itself; tests instantiate or subclass it."""
    return FakeMemory


@pytest.fixture
def perf_clock(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Deterministic stand-in for perf_counter; tests advance state["now"]."""
    state = {"now": 0.0}
    monkeypatch.setattr("amb.callbacks.time.time.perf_counter", lambda: state["now"])
    return state


@pytest.fixture
def stderr_stream(monkeypatch: pytest.MonkeyPatch):
    """Replace sys.stderr with a StringIO so configure() binds its sink to it.

    Teardown restores the real stderr and re-runs configure() so the module's
    import-time default (warnings to stderr) holds for the rest of the suite.

    Yields:
        The StringIO standing in for sys.stderr.
    """
    stream = io.StringIO()
    original = sys.stderr
    monkeypatch.setattr(sys, "stderr", stream)
    yield stream
    sys.stderr = original
    logs.configure()


# One LongMemEval question over a two-session haystack: the first session
# holds the evidence turn, the second is pure noise.
_LONGMEMEVAL_RICH_RECORD: dict = {
    "question_id": 42,
    "question_type": "single-session-user",
    "question": "What instrument did the user pick up?",
    "answer": 1987,
    "question_date": "2023/05/20 (Sat) 02:21",
    "haystack_session_ids": ["answer_s1", "noise_s2"],
    "haystack_dates": ["2023/05/01 (Mon) 10:00", "2023/05/03 (Wed) 18:45"],
    "answer_session_ids": ["answer_s1"],
    "haystack_sessions": [
        [
            {
                "role": "user",
                "content": "I started cello lessons this week.",
                "has_answer": True,
            },
            {"role": "assistant", "content": "Congratulations on the cello!"},
        ],
        [
            {"role": "user", "content": "Suggest a quick pasta dinner."},
            {
                "role": "assistant",
                "content": "Cacio e pepe takes fifteen minutes.",
                "has_answer": False,
            },
        ],
    ],
}


@pytest.fixture
def longmemeval_rich_record() -> dict:
    """A fresh copy of the rich two-session LongMemEval record."""
    return copy.deepcopy(_LONGMEMEVAL_RICH_RECORD)


@pytest.fixture
def longmemeval_rich_sample(tmp_path: Path, longmemeval_rich_record: dict) -> Sample:
    """The rich record, loaded through an offline LongMemEvalLoader."""
    data = tmp_path / "longmemeval_records.json"
    data.write_text(json.dumps([longmemeval_rich_record]))

    class OfflineLoader(LongMemEvalLoader):
        def pull(self, variant: str | None = None) -> Path:
            return data

    return OfflineLoader(cache_dir=tmp_path / "cache").load()[0]
