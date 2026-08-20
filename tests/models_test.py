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

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from amb.contracts.document import Block, Figure, Heading, Paragraph, Rule, Table
from amb.contracts.memory import MemoryHit
from amb.contracts.plot import Point, Series
from amb.contracts.run import IngestionRecord, QuestionRecord, Run


def make_run() -> Run:
    return Run(run_id="r1", system="naive", dataset="locomo")


def test_run_defaults():
    run = make_run()
    assert run.mode == "direct"
    assert run.k == 10
    assert run.system_params == {}
    assert run.question_records == []
    assert run.ingestion_records == []
    assert run.ingestion_model is None
    assert run.variant is None
    assert run.max_turns is None


def test_run_record_lists_are_per_instance():
    first = Run(run_id="a", system="s", dataset="d")
    second = Run(run_id="b", system="s", dataset="d")
    first.add_question({"sample_id": "s1", "question_id": "q1"})
    first.add_ingestion({"sample_id": "s1", "ingest_s": 1.0, "num_turns": 2})
    assert second.question_records == []
    assert second.ingestion_records == []


def test_run_requires_identity_fields():
    with pytest.raises(ValidationError, match="dataset"):
        Run.model_validate({"run_id": "r1", "system": "naive"})


def test_add_question_validates_and_stores_a_dict_row():
    run = make_run()
    record = run.add_question(
        {
            "sample_id": "s1",
            "question_id": "q1",
            "question": "Who moved to Paris?",
            "search_s": 0.25,
            "num_hits": 3,
            "retrieved_session_ids": ["sess-1", "sess-2"],
            "judge_correct": True,
        }
    )
    assert isinstance(record, QuestionRecord)
    assert record.sample_id == "s1"
    assert record.question_id == "q1"
    assert record.search_s == 0.25
    assert record.num_hits == 3
    assert record.retrieved_session_ids == ["sess-1", "sess-2"]
    assert record.judge_correct is True
    assert run.question_records == [record]


def test_add_question_defaults_unset_observations_to_none():
    record = make_run().add_question({"sample_id": "s1", "question_id": "q1"})
    assert record.question is None
    assert record.gold_answer is None
    assert record.search_s is None
    assert record.retrieved_session_ids is None
    assert record.predicted_answer is None
    assert record.judge_correct is None
    assert record.error is None


def test_add_question_accepts_an_already_built_record():
    run = make_run()
    record = QuestionRecord(sample_id="s1", question_id="q1")
    stored = run.add_question(record)
    assert stored is record
    assert run.question_records == [record]


def test_add_question_appends_in_order():
    run = make_run()
    run.add_question({"sample_id": "s1", "question_id": "q1"})
    run.add_question({"sample_id": "s1", "question_id": "q2"})
    assert [r.question_id for r in run.question_records] == ["q1", "q2"]


@pytest.mark.parametrize(
    "row",
    [
        {"question_id": "q1"},  # missing sample_id
        {"sample_id": "s1"},  # missing question_id
        {"sample_id": "s1", "question_id": "q1", "search_s": "fast"},
        {"sample_id": "s1", "question_id": "q1", "num_hits": 2.5},
        {"sample_id": "s1", "question_id": "q1", "judge_correct": "definitely"},
        {"sample_id": "s1", "question_id": "q1", "retrieved_session_ids": "sess-1"},
    ],
)
def test_add_question_invalid_rows_raise_and_do_not_land(row):
    run = make_run()
    with pytest.raises(ValidationError):
        run.add_question(row)
    assert run.question_records == []


def test_add_question_drops_unknown_keys():
    record = make_run().add_question(
        {"sample_id": "s1", "question_id": "q1", "bogus": 1}
    )
    assert not hasattr(record, "bogus")


def test_add_ingestion_validates_and_stores_a_stats_dict():
    run = make_run()
    record = run.add_ingestion(
        {"sample_id": "s1", "ingest_s": 12.5, "num_turns": 40, "num_sessions": 4}
    )
    assert isinstance(record, IngestionRecord)
    assert record.sample_id == "s1"
    assert record.ingest_s == 12.5
    assert record.num_turns == 40
    assert record.num_sessions == 4
    assert record.system_stats == {}
    assert record.token_usage is None
    assert record.num_writes is None
    assert run.ingestion_records == [record]


def test_add_ingestion_accepts_an_already_built_record():
    run = make_run()
    record = IngestionRecord(sample_id="s1", ingest_s=0.5, num_turns=1)
    stored = run.add_ingestion(record)
    assert stored is record
    assert run.ingestion_records == [record]


@pytest.mark.parametrize(
    "stats",
    [
        {"sample_id": "s1", "num_turns": 4},  # missing ingest_s
        {"sample_id": "s1", "ingest_s": 1.0},  # missing num_turns
        {"ingest_s": 1.0, "num_turns": 4},  # missing sample_id
        {"sample_id": "s1", "ingest_s": "slow", "num_turns": 4},
        {"sample_id": "s1", "ingest_s": 1.0, "num_turns": 3.5},
        {"sample_id": "s1", "ingest_s": 1.0, "num_turns": 4, "system_stats": "n/a"},
    ],
)
def test_add_ingestion_invalid_stats_raise_and_do_not_land(stats):
    run = make_run()
    with pytest.raises(ValidationError):
        run.add_ingestion(stats)
    assert run.ingestion_records == []


def test_run_round_trips_through_model_dump():
    run = Run(run_id="r1", system="naive", dataset="locomo", k=5, mode="agentic")
    run.add_question({"sample_id": "s1", "question_id": "q1", "predicted_answer": "42"})
    run.add_ingestion({"sample_id": "s1", "ingest_s": 1.0, "num_turns": 2})
    rebuilt = Run.model_validate(run.model_dump())
    assert rebuilt == run


def test_memory_hit_defaults():
    hit = MemoryHit(content="Alice moved to Paris in May.")
    assert hit.content == "Alice moved to Paris in May."
    assert hit.score is None
    assert hit.turn_ids == []
    assert hit.session_ids == []
    assert hit.metadata == {}


def test_memory_hit_carries_provenance():
    hit = MemoryHit(
        content="fact",
        score=0.87,
        turn_ids=["t1", "t2"],
        session_ids=["sess-1"],
        metadata={"source": "graph"},
    )
    assert hit.score == 0.87
    assert hit.turn_ids == ["t1", "t2"]
    assert hit.session_ids == ["sess-1"]
    assert hit.metadata == {"source": "graph"}


def test_memory_hit_requires_content():
    with pytest.raises(ValidationError, match="content"):
        MemoryHit.model_validate({"score": 0.5})


def test_memory_hit_coerces_integer_score_to_float():
    hit = MemoryHit(content="c", score=1)
    assert hit.score == 1.0
    assert isinstance(hit.score, float)


def test_memory_hit_default_lists_are_per_instance():
    first = MemoryHit(content="a")
    second = MemoryHit(content="b")
    first.turn_ids.append("t1")
    first.session_ids.append("s1")
    assert second.turn_ids == []
    assert second.session_ids == []


def test_heading_fields():
    heading = Heading(level=2, text="Results")
    assert heading.level == 2
    assert heading.text == "Results"


def test_table_keeps_preformatted_cells():
    table = Table(
        header=["system", "score"], rows=[["mem0", "0.61"], ["naive", "0.55"]]
    )
    assert list(table.header) == ["system", "score"]
    assert [list(row) for row in table.rows] == [["mem0", "0.61"], ["naive", "0.55"]]


def test_table_rejects_a_bare_string_header():
    with pytest.raises(ValidationError, match="header"):
        Table.model_validate({"header": "systemscore", "rows": []})


def test_figure_coerces_string_path():
    figure = Figure.model_validate({"alt": "latency vs k", "path": "plots/latency.png"})
    assert figure.path == Path("plots/latency.png")
    assert isinstance(figure.path, Path)


def test_rule_has_no_fields():
    assert Rule() == Rule()
    assert Rule().model_dump() == {}


@pytest.mark.parametrize(
    "payload, expected_type",
    [
        ({"level": 1, "text": "Report"}, Heading),
        ({"text": "Some prose."}, Paragraph),
        ({"header": ["a"], "rows": [["1"]]}, Table),
        ({"alt": "fig", "path": "fig.png"}, Figure),
        ({}, Rule),
    ],
)
def test_block_union_discriminates_by_shape(payload, expected_type):
    block = TypeAdapter(Block).validate_python(payload)
    assert type(block) is expected_type


def test_block_union_round_trips_a_document():
    blocks: list[Block] = [
        Heading(level=1, text="Report"),
        Paragraph(text="Intro."),
        Table(header=["k"], rows=[["10"]]),
        Figure(alt="plot", path=Path("plot.png")),
        Rule(),
    ]
    adapter = TypeAdapter(list[Block])
    assert adapter.validate_python(adapter.dump_python(blocks)) == blocks


def test_point_fields():
    point = Point(label="mem0", x=0.61, y=2.4)
    assert point.label == "mem0"
    assert point.x == 0.61
    assert point.y == 2.4


def test_point_requires_both_coordinates():
    with pytest.raises(ValidationError, match="Field required"):
        Point.model_validate({"label": "mem0", "x": 0.61})


def test_point_rejects_non_numeric_coordinates():
    with pytest.raises(ValidationError, match="valid number"):
        Point.model_validate({"label": "mem0", "x": "far", "y": 1.0})


def test_series_defaults_to_empty_axes():
    series = Series(label="naive")
    assert series.xs == []
    assert series.ys == []


def test_series_coerces_integer_ks_to_float():
    series = Series(label="mem0", xs=[1, 5, 10], ys=[0.4, 0.5, 0.6])
    assert series.xs == [1.0, 5.0, 10.0]
    assert all(isinstance(x, float) for x in series.xs)
    assert series.ys == [0.4, 0.5, 0.6]


def test_series_rejects_non_numeric_axis_values():
    with pytest.raises(ValidationError, match="valid number"):
        Series.model_validate({"label": "s", "xs": ["k=1"], "ys": []})


def test_series_default_axes_are_per_instance():
    first = Series(label="a")
    second = Series(label="b")
    first.xs.append(1.0)
    first.ys.append(0.5)
    assert second.xs == []
    assert second.ys == []
