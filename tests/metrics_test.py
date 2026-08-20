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

from amb.base.metric import Metric, Scorer
from amb.contracts.run import IngestionRecord, QuestionRecord
from amb.metrics import (
    AnswerF1,
    Count,
    DictSum,
    ExactMatch,
    LatencyPercentiles,
    Mean,
    RetrievalF1,
    RetrievalMetric,
    RetrievalPrecision,
    RetrievalRecall,
    Sum,
    TurnF1,
    TurnPrecision,
    TurnRecall,
    ValueCounts,
    default_category_metrics,
    default_metrics,
    normalize_answer,
)


def _q(**fields: Any) -> QuestionRecord:
    """A question record carrying only the fields the test cares about."""
    return QuestionRecord(sample_id="s1", question_id="q1", **fields)


def _ing(ingest_s: float = 0.0, num_turns: int = 0, **fields: Any) -> IngestionRecord:
    """An ingestion record carrying only the fields the test cares about."""
    return IngestionRecord(
        sample_id="s1", ingest_s=ingest_s, num_turns=num_turns, **fields
    )


# --- the Metric base lifecycle -------------------------------------------------


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


# --- Mean ----------------------------------------------------------------------


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


# --- Sum -----------------------------------------------------------------------


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


# --- DictSum -------------------------------------------------------------------


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


# --- Count ---------------------------------------------------------------------


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


# --- ValueCounts ---------------------------------------------------------------


def test_value_counts_per_distinct_value() -> None:
    metric = ValueCounts("categories", "category")
    for category in ("b", "a", "b"):
        metric.update_state(_q(category=category))
    assert metric.result() == {"a": 1, "b": 2}
    assert list(metric.result()) == ["a", "b"]


def test_value_counts_stringifies_values() -> None:
    metric = ValueCounts("hits", "num_hits")
    for hits in (1, 2, 1):
        metric.update_state(_q(num_hits=hits))
    assert metric.result() == {"1": 2, "2": 1}


def test_value_counts_reset_forgets_everything() -> None:
    metric = ValueCounts("categories", "category")
    metric.update_state(_q(category="a"))
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == {}


# --- LatencyPercentiles --------------------------------------------------------


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


# --- normalize_answer ----------------------------------------------------------


def test_normalize_answer_squad_convention() -> None:
    assert normalize_answer("The Quick,  Brown Fox!") == "quick brown fox"


def test_normalize_answer_strips_articles_after_punctuation() -> None:
    # punctuation removal runs first, so "a.n" collapses to the article "an"
    assert normalize_answer("a.n answer") == "answer"


def test_normalize_answer_all_punctuation_is_empty() -> None:
    assert normalize_answer("...") == ""


# --- AnswerF1 / ExactMatch scoring ---------------------------------------------


def test_answer_f1_identical_after_normalization() -> None:
    assert AnswerF1().score("The Ramen!", "ramen") == 1.0


def test_answer_f1_disjoint_tokens() -> None:
    assert AnswerF1().score("cat", "dog") == 0.0


def test_answer_f1_partial_overlap() -> None:
    # pred {cat, sat}: precision 2/2, recall 2/3 -> f1 0.8
    assert AnswerF1().score("the cat sat", "cat sat down") == pytest.approx(0.8)


def test_answer_f1_counts_token_multiplicity() -> None:
    # overlap min(2,1)=1: precision 1/2, recall 1/1 -> f1 2/3
    assert AnswerF1().score("dog dog", "dog") == pytest.approx(2 / 3)


def test_answer_f1_both_sides_normalize_to_empty() -> None:
    assert AnswerF1().score("the", "a") == 1.0


def test_answer_f1_empty_prediction_nonempty_gold() -> None:
    assert AnswerF1().score("", "cat") == 0.0


def test_answer_f1_reports_under_its_dotted_free_name() -> None:
    assert AnswerF1().name == "answer_f1"


def test_exact_match_ignores_form() -> None:
    metric = ExactMatch()
    assert metric.name == "exact_match"
    assert metric.score("The Answer!", "answer") == 1.0


def test_exact_match_different_answers() -> None:
    assert ExactMatch().score("42", "43") == 0.0


# --- the Scorer guard and macro-averaging --------------------------------------


def test_scorer_default_pair_keys() -> None:
    assert Scorer.predicted_key == "predicted_answer"
    assert Scorer.gold_key == "gold_answer"


def test_scorer_skips_missing_prediction() -> None:
    metric = AnswerF1()
    metric.update_state(_q(gold_answer="ramen"))
    assert metric.count == 0
    assert metric.result() == 0.0


@pytest.mark.parametrize("gold", [None, ""])
def test_scorer_skips_empty_gold(gold) -> None:
    metric = AnswerF1()
    metric.update_state(_q(predicted_answer="ramen", gold_answer=gold))
    assert metric.count == 0


def test_scorer_counts_empty_prediction_as_zero() -> None:
    # "" is a prediction (just a wrong one), unlike a missing one
    metric = AnswerF1()
    metric.update_state(_q(predicted_answer="", gold_answer="ramen"))
    assert metric.count == 1
    assert metric.result() == 0.0


def test_answer_f1_macro_averages_across_records() -> None:
    metric = AnswerF1()
    metric.update_state(_q(predicted_answer="ramen", gold_answer="Ramen!"))  # 1.0
    metric.update_state(
        _q(predicted_answer="the cat sat", gold_answer="cat sat down")
    )  # 0.8
    assert metric.count == 2
    assert metric.result() == pytest.approx(0.9)
    metric.reset_state()
    assert metric.count == 0
    assert metric.result() == 0.0


# --- retrieval and turn precision/recall/f1 ------------------------------------


def test_retrieval_scores_hand_computed() -> None:
    scores = RetrievalMetric.scores(["s1", "s2", "s3", "s4"], ["s2", "s4", "s5"])
    assert scores["retrieval_precision"] == 0.5
    assert scores["retrieval_recall"] == pytest.approx(2 / 3)
    assert scores["retrieval_f1"] == pytest.approx(4 / 7)


def test_retrieval_scores_empty_retrieved() -> None:
    scores = RetrievalMetric.scores([], ["s1"])
    assert scores == {
        "retrieval_precision": 0.0,
        "retrieval_recall": 0.0,
        "retrieval_f1": 0.0,
    }


def test_retrieval_scores_empty_evidence() -> None:
    scores = RetrievalMetric.scores(["s1"], [])
    assert scores == {
        "retrieval_precision": 0.0,
        "retrieval_recall": 0.0,
        "retrieval_f1": 0.0,
    }


def test_retrieval_scores_deduplicate_ids() -> None:
    # sets, not lists: a doubly-retrieved id counts once
    scores = RetrievalMetric.scores(["a", "a", "b"], ["a"])
    assert scores["retrieval_precision"] == 0.5
    assert scores["retrieval_recall"] == 1.0


def test_retrieval_precision_macro_average_and_guards() -> None:
    metric = RetrievalPrecision()
    metric.update_state(
        _q(retrieved_session_ids=["s1", "s2"], evidence_session_ids=["s2"])
    )  # 0.5
    metric.update_state(
        _q(retrieved_session_ids=["s3"], evidence_session_ids=["s3"])
    )  # 1.0
    metric.update_state(_q())  # no retrieval fields: skipped
    metric.update_state(
        _q(retrieved_session_ids=["s9"], evidence_session_ids=[])
    )  # empty gold: skipped
    assert metric.count == 2
    assert metric.result() == 0.75


def test_retrieval_recall_counts_empty_retrieval_as_zero() -> None:
    # [] is a (failed) retrieval, unlike a missing field: it scores 0.0
    metric = RetrievalRecall()
    metric.update_state(_q(retrieved_session_ids=[], evidence_session_ids=["s1"]))
    assert metric.count == 1
    assert metric.result() == 0.0


def test_retrieval_f1_single_record() -> None:
    metric = RetrievalF1()
    metric.update_state(
        _q(
            retrieved_session_ids=["s1", "s2", "s3", "s4"],
            evidence_session_ids=["s2", "s4", "s5"],
        )
    )
    assert metric.result() == pytest.approx(4 / 7)


def test_turn_precision_reads_turn_ids_not_session_ids() -> None:
    metric = TurnPrecision()
    metric.update_state(
        _q(
            retrieved_turn_ids=["t1", "t2", "t3"],
            evidence_turn_ids=["t1"],
            retrieved_session_ids=["s1"],  # would score 1.0 if wrongly read
            evidence_session_ids=["s1"],
        )
    )
    assert metric.result() == pytest.approx(1 / 3)


def test_turn_metrics_skip_session_only_records() -> None:
    metric = TurnRecall()
    metric.update_state(_q(retrieved_session_ids=["s1"], evidence_session_ids=["s1"]))
    assert metric.count == 0
    assert metric.result() == 0.0


@pytest.mark.parametrize(
    "cls, name, predicted_key, gold_key, component",
    [
        (
            RetrievalPrecision,
            "retrieval_precision",
            "retrieved_session_ids",
            "evidence_session_ids",
            "retrieval_precision",
        ),
        (
            RetrievalRecall,
            "retrieval_recall",
            "retrieved_session_ids",
            "evidence_session_ids",
            "retrieval_recall",
        ),
        (
            RetrievalF1,
            "retrieval_f1",
            "retrieved_session_ids",
            "evidence_session_ids",
            "retrieval_f1",
        ),
        (
            TurnPrecision,
            "turn_precision",
            "retrieved_turn_ids",
            "evidence_turn_ids",
            "retrieval_precision",
        ),
        (
            TurnRecall,
            "turn_recall",
            "retrieved_turn_ids",
            "evidence_turn_ids",
            "retrieval_recall",
        ),
        (
            TurnF1,
            "turn_f1",
            "retrieved_turn_ids",
            "evidence_turn_ids",
            "retrieval_f1",
        ),
    ],
)
def test_retrieval_metric_wiring(cls, name, predicted_key, gold_key, component) -> None:
    assert cls().name == name
    assert cls.predicted_key == predicted_key
    assert cls.gold_key == gold_key
    assert cls.component == component


# --- the default metric sets ---------------------------------------------------


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


def test_default_metrics_cover_both_retrieval_levels() -> None:
    names = {m.name for m in default_metrics()}
    assert {"retrieval_precision", "retrieval_recall", "retrieval_f1"} <= names
    assert {"turn_precision", "turn_recall", "turn_f1"} <= names
    assert {"answer_f1", "exact_match"} <= names


def test_category_metrics_names() -> None:
    names = [m.name for m in default_category_metrics()]
    assert names == [
        "retrieval_precision",
        "retrieval_recall",
        "retrieval_f1",
        "answer_f1",
        "exact_match",
        "judge_accuracy",
    ]


def test_category_metrics_are_fresh_instances_per_call() -> None:
    first = default_category_metrics()
    first[-1].update_state(_q(judge_correct=True))
    second = default_category_metrics()
    assert first[-1].count == 1
    assert second[-1].count == 0


def test_category_metrics_are_row_level_only() -> None:
    # categories are a property of questions, so no ingestion sections here
    for metric in default_category_metrics():
        assert "." not in metric.name
