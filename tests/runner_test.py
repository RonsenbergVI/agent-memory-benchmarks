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

"""The run loop's fault handling: one sample's failure never costs the rest."""

import time

import pytest

from amb.base import Benchmark, Memory
from amb.contracts import Conversation, MemoryHit, QAPair, Sample, Session, Turn
from amb.reporting import RunReport
from amb.runner import RunConfig, Runner


class BrittleMemory(Memory):
    """Stores nothing; fails in the way its conversation id asks for."""

    name = "brittle"
    description = "in-test memory that fails on demand"

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        if conversation_id.startswith("ingest-fails"):
            raise RuntimeError("extractor down")
        if conversation_id.startswith("interrupt"):
            # let the other samples finish first, so no thread outlives the test
            time.sleep(0.5)
            raise KeyboardInterrupt

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        return [MemoryHit(content="a fact", session_ids=["s1"])]


def _sample(conversation_id: str, questions: int = 2) -> Sample:
    session = Session(
        session_id="s1",
        turns=[Turn(turn_id="t1", speaker="user", text="hello")],
    )
    return Sample(
        sample_id=conversation_id,
        dataset="unit",
        conversation=Conversation(conversation_id=conversation_id, sessions=[session]),
        qa=[
            QAPair(
                question_id=f"{conversation_id}:{i}",
                question="what was said?",
                answer="hello",
                evidence_session_ids=["s1"],
            )
            for i in range(questions)
        ],
    )


def _run(samples: list[Sample], workers: int = 1, memory: type[Memory] = BrittleMemory):
    config = RunConfig(dataset="unit", model="test", workers=workers)
    return Runner(Benchmark.for_system(memory)(), config, samples=samples).run()


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failed_sample_is_recorded_and_the_rest_still_run(workers):
    run = _run([_sample("fine-1"), _sample("ingest-fails"), _sample("fine-2")], workers)
    by_id = {record.sample_id: record for record in run.ingestion_records}
    assert by_id["ingest-fails"].error == "RuntimeError: extractor down"
    assert by_id["fine-1"].error is None
    assert by_id["fine-2"].error is None
    asked = [row.sample_id for row in run.question_records]
    assert asked.count("ingest-fails") == 0
    assert asked.count("fine-1") == 2
    assert asked.count("fine-2") == 2
    # the dirty-run marker the summary carries
    assert RunReport(run).summary()["errors"] == 1


def test_a_setup_failure_lands_as_an_empty_errored_record():
    class SetupFails(BrittleMemory):
        def setup(self) -> None:
            raise ConnectionError("no database")

    run = _run([_sample("c1")], memory=SetupFails)
    [record] = run.ingestion_records
    assert record.error == "ConnectionError: no database"
    assert record.num_turns == 0
    assert run.question_records == []


def test_an_interrupt_keeps_finished_samples_and_marks_the_rest():
    run = _run([_sample("fine-1"), _sample("interrupt"), _sample("fine-2")])
    by_id = {record.sample_id: record for record in run.ingestion_records}
    assert by_id["fine-1"].error is None
    assert str(by_id["interrupt"].error).startswith("KeyboardInterrupt")
    assert str(by_id["fine-2"].error).startswith("KeyboardInterrupt")
    assert [row.sample_id for row in run.question_records] == ["fine-1", "fine-1"]
    assert RunReport(run).summary()["errors"] == 2


def test_an_interrupt_under_workers_merges_only_finished_samples():
    # fine-1 completes before the interrupt is seen: results come back in order
    run = _run([_sample("fine-1"), _sample("interrupt"), _sample("fine-2")], workers=3)
    by_id = {record.sample_id: record for record in run.ingestion_records}
    assert set(by_id) == {"fine-1", "interrupt", "fine-2"}
    assert by_id["fine-1"].error is None
    assert str(by_id["interrupt"].error).startswith("KeyboardInterrupt")
    assert [row.sample_id for row in run.question_records].count("fine-1") == 2
    assert RunReport(run).summary()["errors"] >= 1
