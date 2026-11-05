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

"""Judging: verdicts land per row, a failed verdict skips its row, runs save first."""

import json

import pytest
from click.testing import CliRunner

from amb.cli.main import cli
from amb.contracts import (
    Conversation,
    Judgment,
    QAPair,
    QuestionRecord,
    Run,
    Sample,
    Session,
    Turn,
)
from amb.reporting import RunReport
from amb.runner import Runner


def _run(n: int = 4) -> Run:
    return Run(
        run_id="20260101T000000Z-abc123",
        system="fake",
        dataset="unit",
        model="test",
        question_records=[
            QuestionRecord(
                sample_id="s",
                question_id=f"q{i}",
                question=f"q{i}?",
                gold_answer="gold",
                predicted_answer=f"pred {i}",
            )
            for i in range(n)
        ],
    )


@pytest.fixture
def fake_judge(monkeypatch):
    """A judge that grades by prediction and always fails on 'pred 2'."""
    calls: list[str] = []

    def judge_answer(question, prediction, gold, model):
        calls.append(prediction)
        if prediction == "pred 2":
            raise RuntimeError("rate limited")
        return Judgment(correct=prediction.endswith("1"), reasoning="because")

    monkeypatch.setattr("amb.agent.judge_answer", judge_answer)
    return calls


@pytest.mark.parametrize("workers", [1, 3])
def test_a_failed_verdict_leaves_its_row_unjudged_and_the_rest_graded(
    fake_judge, workers
):
    report = RunReport(_run())
    assert report.judge("test", workers=workers) == 3
    verdicts = {
        row.question_id: row.judge_correct for row in report.run.question_records
    }
    assert verdicts == {"q0": False, "q1": True, "q2": None, "q3": False}
    assert report.run.judge_model == "test"


def test_unjudged_rows_are_the_only_ones_a_second_pass_grades(fake_judge):
    report = RunReport(_run())
    report.judge("test")
    assert report.judge("test") == 0
    assert fake_judge.count("pred 2") == 2
    assert fake_judge.count("pred 0") == 1


def test_rows_without_gold_or_prediction_are_skipped(fake_judge):
    run = _run(2)
    run.question_records[0].gold_answer = None
    run.question_records[1].predicted_answer = None
    assert RunReport(run).judge("test") == 0
    assert fake_judge == []


def test_run_saves_before_judging(tmp_path, monkeypatch):
    """A judge that dies after the run still leaves the run on disk."""
    sample = Sample(
        sample_id="c1",
        dataset="locomo",
        conversation=Conversation(
            conversation_id="c1",
            sessions=[
                Session(
                    session_id="s1",
                    turns=[Turn(turn_id="t1", speaker="user", text="hi there")],
                )
            ],
        ),
        qa=[QAPair(question_id="q", question="what was said?", answer="hi there")],
    )
    monkeypatch.setattr(Runner, "load_samples", lambda self: [sample])

    def boom(self, model, force=False, workers=1):
        raise RuntimeError("judge down")

    monkeypatch.setattr(RunReport, "judge", boom)
    result = CliRunner().invoke(
        cli,
        ["run", "--system", "naive", "--dataset", "locomo", "--model", "test"]
        + ["--judge", "--runs-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    [results] = list(tmp_path.rglob("results.jsonl"))
    assert json.loads(results.read_text().splitlines()[0])["predicted_answer"]


def test_judge_command_keeps_the_verdicts_of_a_pass_that_died(tmp_path, monkeypatch):
    run_dir = RunReport(_run(2)).save(tmp_path)

    def partial(self, model, force=False, workers=1):
        self.run.question_records[0].judge_correct = True
        raise RuntimeError("judge down")

    monkeypatch.setattr(RunReport, "judge", partial)
    result = CliRunner().invoke(cli, ["judge", str(run_dir)])
    assert result.exit_code == 1
    rows = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text().splitlines()
    ]
    assert rows[0]["judge_correct"] is True
    assert "judge_correct" not in rows[1]
