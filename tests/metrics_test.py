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

"""Metrics: the stateful lifecycle, hand-computed scores, and the default sets."""

from typing import Any

import pytest

from amb.base.metric import Metric
from amb.contracts.run import IngestionRecord, QuestionRecord
from amb.metrics import (
    Attribution,
    Count,
    DictSum,
    LatencyPercentiles,
    Mean,
    MemoryLift,
    Sum,
    default_category_metrics,
    default_metrics,
)


def _q(**fields: Any) -> QuestionRecord:
    """A question record carrying only the fields the test cares about."""
    return QuestionRecord(sample_id="s1", question_id="q1", **fields)


def _ing(ingest_s: float = 0.0, num_turns: int = 0, **fields: Any) -> IngestionRecord:
    """An ingestion record carrying only the fields the test cares about."""
    return IngestionRecord(
        sample_id="s1", ingest_s=ingest_s, num_turns=num_turns, **fields
    )


def test_key_defaults_to_name() -> None:
    metric = Sum("num_turns")
    assert metric.key == "num_turns"


def test_explicit_key_overrides_name() -> None:
    metric = Sum("ingest.total_s", "ingest_s")
    assert metric.name == "ingest.total_s"
    assert metric.key == "ingest_s"


def test_records_without_the_field_are_ignored() -> None:
    metric = Sum("total", "not_a_record_field")
    metric.update_state(_q())
    assert metric.count == 0
    assert metric.result() == 0


def test_none_valued_fields_are_ignored() -> None:
    metric = Mean("tokens.input_per_question", "input_tokens")
    metric.update_state(_q(input_tokens=10))
    metric.update_state(_q())  # input_tokens defaults to None
    metric.update_state(_q(input_tokens=20))
    assert metric.count == 2
    assert metric.result() == 15.0


def test_report_when_empty_defaults() -> None:
    # an error count of 0 is a statement, so Count opts in; the rest do not
    assert Metric.report_when_empty is False
    assert Sum.report_when_empty is False
    assert Mean.report_when_empty is False
    assert Count.report_when_empty is True


def test_mean_hand_computed() -> None:
    metric = Mean("mean_search", "search_s")
    for latency in (1.0, 2.0, 6.0):
        metric.update_state(_q(search_s=latency))
    assert metric.count == 3
    assert metric.result() == 3.0


def test_mean_is_zero_when_nothing_seen() -> None:
    assert Mean("mean_search", "search_s").result() == 0.0


def test_mean_reset_starts_a_new_scope() -> None:
    metric = Mean("mean_search", "search_s")
    metric.update_state(_q(search_s=2.0))
    metric.update_state(_q(search_s=4.0))
    assert metric.result() == 3.0
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == 0.0
    metric.update_state(_q(search_s=8.0))
    assert metric.result() == 8.0


def test_mean_counts_falsy_but_present_booleans() -> None:
    metric = Mean("judge_accuracy", "judge_correct")
    for verdict in (True, False, True):
        metric.update_state(_q(judge_correct=verdict))
    assert metric.count == 3
    assert metric.result() == pytest.approx(2 / 3)


def test_sum_hand_computed() -> None:
    metric = Sum("ingest.total_s", "ingest_s")
    metric.update_state(_ing(ingest_s=1.5))
    metric.update_state(_ing(ingest_s=2.5))
    assert metric.count == 2
    assert metric.result() == 4.0


def test_sum_reset_starts_a_new_scope() -> None:
    metric = Sum("ingest.total_turns", "num_turns")
    metric.update_state(_ing(num_turns=7))
    metric.update_state(_ing(num_turns=3))
    assert metric.result() == 10
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == 0
    metric.update_state(_ing(num_turns=4))
    assert metric.result() == 4


def test_dictsum_merges_dicts_keywise() -> None:
    metric = DictSum("memory_tokens.ingest", "token_usage")
    metric.update_state(_ing(token_usage={"input": 10, "output": 5}))
    metric.update_state(_ing(token_usage={"output": 7, "cache": 1}))
    assert metric.count == 2
    assert metric.result() == {"cache": 1, "input": 10, "output": 12}


def test_dictsum_result_is_key_sorted() -> None:
    metric = DictSum("memory_tokens.ingest", "token_usage")
    metric.update_state(_ing(token_usage={"zebra": 1, "alpha": 2}))
    assert list(metric.result()) == ["alpha", "zebra"]


def test_dictsum_reset_clears_totals() -> None:
    metric = DictSum("memory_tokens.ingest", "token_usage")
    metric.update_state(_ing(token_usage={"input": 10}))
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == {}


def test_count_counts_records_carrying_the_field() -> None:
    metric = Count("errors", "error")
    metric.update_state(_q(error="boom"))
    metric.update_state(_q())  # no error: not counted
    metric.update_state(_q(error="crash"))
    assert metric.result() == 2


def test_count_accumulate_is_a_noop() -> None:
    # counting happens in update_state; accumulate alone must not count
    metric = Count("errors", "error")
    metric.accumulate("boom")
    assert metric.result() == 0


def test_count_reset_returns_to_zero() -> None:
    metric = Count("errors", "error")
    metric.update_state(_q(error="boom"))
    metric.reset_state()
    assert metric.result() == 0


def test_percentiles_nearest_rank_hand_computed() -> None:
    # sorted sample [0.1..0.5]: p50 -> index round(.5*4)=2, p95/p99 -> index 4
    metric = LatencyPercentiles("search_latency", "search_s")
    for latency in (0.5, 0.1, 0.4, 0.2, 0.3):
        metric.update_state(_q(search_s=latency))
    assert metric.result() == {"p50_s": 0.3, "p95_s": 0.5, "p99_s": 0.5}


def test_percentiles_empty_sample_is_zero() -> None:
    metric = LatencyPercentiles("search_latency", "search_s")
    assert metric.result() == {"p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}


def test_percentiles_custom_points() -> None:
    metric = LatencyPercentiles("search_latency", "search_s", points=(0, 100))
    for latency in (3.0, 1.0, 2.0):
        metric.update_state(_q(search_s=latency))
    assert metric.result() == {"p0_s": 1.0, "p100_s": 3.0}


def test_percentiles_reset_discards_sample() -> None:
    metric = LatencyPercentiles("search_latency", "search_s")
    metric.update_state(_q(search_s=9.0))
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == {"p50_s": 0.0, "p95_s": 0.0, "p99_s": 0.0}


def test_percentile_static_empty_list_is_zero() -> None:
    assert LatencyPercentiles.percentile([], 50) == 0.0


def test_default_metric_names_are_unique() -> None:
    names = [m.name for m in default_metrics()]
    assert len(names) == len(set(names))


def test_dotted_names_split_into_section_and_leaf() -> None:
    dotted = [m.name for m in default_metrics() if "." in m.name]
    assert dotted  # the nested-section convention is in use
    for name in dotted:
        section, _, leaf = name.partition(".")
        assert section
        assert leaf
        assert "." not in leaf


def test_default_metrics_load_bearing_entries() -> None:
    by_name = {m.name: m for m in default_metrics()}

    ingest_total = by_name["ingest.total_s"]
    assert isinstance(ingest_total, Sum)
    assert ingest_total.key == "ingest_s"

    memory_ingest = by_name["memory_tokens.ingest"]
    assert isinstance(memory_ingest, DictSum)
    assert memory_ingest.key == "token_usage"

    memory_search = by_name["memory_tokens.search_total"]
    assert isinstance(memory_search, Sum)
    assert memory_search.key == "memory_tokens"

    search_latency = by_name["search_latency"]
    assert isinstance(search_latency, LatencyPercentiles)
    assert search_latency.key == "search_s"
    assert search_latency.points == (50, 95, 99)

    errors = by_name["errors"]
    assert isinstance(errors, Count)
    assert errors.key == "error"

    judge = by_name["judge_accuracy"]
    assert isinstance(judge, Mean)
    assert judge.key == "judge_correct"

    # undotted on purpose: top-level floats are what the tables pick up
    hits = by_name["hits_per_query"]
    assert type(hits) is Mean
    assert hits.key == "num_hits"


def test_default_metrics_score_no_retrieval_or_overlap() -> None:
    names = {m.name for m in default_metrics()}
    assert "judge_accuracy" in names
    assert not any(
        n.startswith(("retrieval_", "turn_")) or n in {"answer_f1", "exact_match"}
        for n in names
    )


def test_category_metrics_names() -> None:
    names = [m.name for m in default_category_metrics()]
    assert names == [
        "judge_accuracy",
        "judge_accuracy_ci",
        "attribution",
        "memory_lift",
    ]


def test_category_metrics_are_fresh_instances_per_call() -> None:
    first = default_category_metrics()
    first[-1].update_state(_attributed(True, True))
    second = default_category_metrics()
    assert first[-1].count == 1
    assert second[-1].count == 0


def test_category_metrics_are_row_level_only() -> None:
    # categories are a property of questions, so no ingestion sections here
    for metric in default_category_metrics():
        assert "." not in metric.name


def test_wilson_interval_brackets_the_rate_and_stays_in_bounds() -> None:
    from amb.contracts.run import QuestionRecord
    from amb.metrics import WilsonInterval

    metric = WilsonInterval("judge_accuracy_ci", "judge_correct")
    for i in range(100):
        metric.update_state(
            QuestionRecord(sample_id="s", question_id=str(i), judge_correct=i < 80)
        )
    interval = metric.result()
    assert interval["n"] == 100
    assert 0.70 < interval["low"] < 0.80 < interval["high"] < 0.88
    # all-correct still leaves room below 1.0 and never reaches past it
    perfect = WilsonInterval("x", "judge_correct")
    for i in range(10):
        perfect.update_state(
            QuestionRecord(sample_id="s", question_id=str(i), judge_correct=True)
        )
    assert perfect.result()["high"] == 1.0
    assert perfect.result()["low"] < 0.75


def _attributed(found: bool, correct: bool) -> QuestionRecord:
    return _q(
        retrieved_session_ids=["1", "2"] if found else ["2"],
        evidence_session_ids=["1"],
        judge_correct=correct,
    )


def test_attribution_cells_hand_computed() -> None:
    metric = Attribution()
    for found, correct, times in (
        (True, True, 3),
        (True, False, 1),
        (False, True, 1),
        (False, False, 3),
    ):
        for _ in range(times):
            metric.update_state(_attributed(found, correct))
    result = metric.result()
    assert result["n"] == 8
    assert result["correct"] == pytest.approx(3 / 8)
    assert result["agent_fault"] == pytest.approx(1 / 8)
    assert result["lucky"] == pytest.approx(1 / 8)
    assert result["memory_fault"] == pytest.approx(3 / 8)
    assert result["found"] == pytest.approx(0.5)
    assert result["missed"] == pytest.approx(0.5)
    assert result["correct_when_found"] == pytest.approx(0.75)
    assert result["correct_when_missed"] == pytest.approx(0.25)
    assert result["lift"] == pytest.approx(0.5 * (0.75 - 0.25))


def test_attribution_found_means_every_evidence_session() -> None:
    # half the chain retrieved is a miss: multi-hop fails on the missing hop
    metric = Attribution()
    metric.update_state(
        _q(
            retrieved_session_ids=["1", "9"],
            evidence_session_ids=["1", "2"],
            judge_correct=False,
        )
    )
    assert metric.result()["found"] == 0.0


@pytest.mark.parametrize(
    "fields",
    [
        {"evidence_session_ids": ["1"], "judge_correct": True},
        {
            "retrieved_session_ids": ["1"],
            "evidence_session_ids": [],
            "judge_correct": True,
        },
        {"retrieved_session_ids": ["1"], "judge_correct": True},
        {"retrieved_session_ids": ["1"], "evidence_session_ids": ["1"]},
    ],
)
def test_attribution_skips_unlabelled_or_unjudged_rows(fields) -> None:
    # an unlabelled or unjudged question is left out, never a miss
    metric = Attribution()
    metric.update_state(_q(**fields))
    assert metric.count == 0
    assert metric.result()["n"] == 0


def test_attribution_empty_scope_reports_zero_rates() -> None:
    result = Attribution().result()
    assert result["found"] == 0.0
    assert result["missed"] == 0.0
    assert result["correct_when_found"] == 0.0
    assert result["correct_when_missed"] == 0.0


def test_attribution_reset_empties_every_cell() -> None:
    metric = Attribution()
    metric.update_state(_attributed(True, True))
    metric.reset_state()
    metric.update_state(_attributed(False, False))
    result = metric.result()
    assert result["n"] == 1
    assert result["memory_fault"] == 1.0
    assert result["correct"] == 0.0


def test_memory_lift_is_attribution_lift_as_a_top_level_float() -> None:
    lift, cells = MemoryLift(), Attribution()
    for found, correct in ((True, True), (True, False), (False, False), (False, True)):
        for metric in (lift, cells):
            metric.update_state(_attributed(found, correct))
    assert lift.count == 4
    assert lift.result() == pytest.approx(cells.result()["lift"])
    # undotted on purpose: top-level floats are what the tables pick up
    assert "." not in lift.name
    lift.reset_state()
    assert lift.count == 0
    assert lift.result() == 0.0


def test_memory_lift_in_both_metric_sets() -> None:
    assert any(isinstance(m, MemoryLift) for m in default_metrics())
    assert any(isinstance(m, MemoryLift) for m in default_category_metrics())
