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

import pytest


def test_usage_coverage_keeps_three_cost_states_apart():
    """Full, incomplete and unmeasurable must not read as each other.

    hindsight spends inside its own server, so a 0 would sit beside
    naive's genuine zero and mean the opposite. A system that sees only
    part of its own spend is real but short of the truth, so it is
    starred rather than ranked against complete numbers.
    """
    from amb.reporting.run import ComparisonReport

    cell = ComparisonReport._cell
    full = {"system": "fraise", "memory_tokens_total": 130112.0}
    partial = {
        "system": "mem0",
        "memory_tokens_total": 4200.0,
        "usage_coverage": "partial",
    }
    none = {"system": "hindsight", "usage_coverage": "none"}

    assert cell(full, "memory_tokens_total", "{:,.0f}") == "130,112"
    assert cell(partial, "memory_tokens_total", "{:,.0f}") == "4,200*"
    assert cell(none, "memory_tokens_total", "{:,.0f}") == "n/a"
    # full coverage with nothing spent yet is blank, not "n/a"
    assert cell({"system": "x"}, "memory_tokens_total", "{:,.0f}") == ""


def test_summary_markdown_explains_only_the_cost_marks_it_used():
    """The asterisk must say what it means — and not appear unprompted."""
    from amb.reporting.run import ComparisonReport

    notes = ComparisonReport._usage_notes
    assert notes(["130,112", "4,200*"]).startswith("`*`")
    assert "incomplete" in notes(["4,200*"])
    assert "not measurable" in notes(["n/a"])
    # a table where every system is fully accounted carries no footnote
    assert notes(["130,112", "8,154", ""]) == ""


def _categorised(
    system: str, version: str, k: int, accuracy: float, cats: dict
) -> dict:
    return {
        "run_id": f"20260901T000000Z-{system}",
        "system": system,
        "system_version": version,
        "dataset": "locomo",
        "k": k,
        "judge_accuracy": accuracy,
        "by_category": {c: {"judge_accuracy": v} for c, v in cats.items()},
    }


def test_category_table_bolds_the_best_in_each_column():
    from amb.reporting.run import ComparisonReport

    report = ComparisonReport(
        [
            _categorised(
                "alpha", "1.0", 10, 0.60, {"multi-hop": 0.30, "temporal": 0.90}
            ),
            _categorised(
                "beta", "2.0", 10, 0.50, {"multi-hop": 0.70, "temporal": 0.10}
            ),
        ]
    )
    header, rows = report.category_table(10)
    # columns are discovered from the runs, not hardcoded per dataset
    assert header == ["system", "version", "multi-hop", "temporal", "overall"]
    # best overall first, and each column bolded independently of the ranking
    assert rows[0] == ["alpha", "1.0", "0.300", "**0.900**", "**0.600**"]
    assert rows[1] == ["beta", "2.0", "**0.700**", "0.100", "0.500"]


def test_category_table_leaves_an_unreported_category_blank():
    from amb.reporting.run import ComparisonReport

    report = ComparisonReport(
        [
            _categorised(
                "alpha", "1.0", 10, 0.60, {"multi-hop": 0.30, "temporal": 0.90}
            ),
            _categorised("beta", "2.0", 10, 0.50, {"multi-hop": 0.70}),
        ]
    )
    _, rows = report.category_table(10)
    # a blank says "not measured"; a 0.000 would rank as "measured and bad"
    assert rows[1][-2] == ""


def test_category_table_is_empty_when_runs_carry_no_categories():
    from amb.reporting.run import ComparisonReport

    plain = {"run_id": "r", "system": "alpha", "dataset": "locomo", "k": 10}
    assert ComparisonReport([plain]).category_table(10) == ([], [])


def test_table_figure_strips_the_emphasis_markers(tmp_path):
    # the same string feeds markdown (where ** is bold) and the figure
    # (where it must render bold, not print asterisks)
    from amb.reporting.plots import _plain, table

    assert _plain("**0.900**") == "0.900"
    assert _plain("0.900") == "0.900"
    assert _plain("") == ""
    path = table(
        ["system", "version", "multi-hop"],
        [["alpha", "1.0", "**0.900**"]],
        output=tmp_path / "t.png",
    )
    assert path.stat().st_size > 0


def test_summary_table_shows_what_each_system_spent_of_k():
    # k is granted; a system returning fewer hits must not read as ranking better
    from amb.reporting.run import ComparisonReport

    spent = {
        "run_id": "20260901T000000Z",
        "system": "alpha",
        "dataset": "locomo",
        "k": 10,
        "judge_accuracy": 0.5,
        "hits_per_query": 5.5745,
    }
    older = {"run_id": "20260801T000000Z", "system": "beta", "dataset": "locomo"}
    header, rows = ComparisonReport([spent, {**older, "k": 10}]).summary_table(10)
    hits = header.index("hits/q")
    assert rows[0][hits] == "5.57"
    # a run scored before the metric existed is blank, not 0.00
    assert rows[1][hits] == ""


def test_variants_get_their_own_sections_and_plot_directories():
    """Different variants are different experiments, never blended."""
    from pathlib import Path

    from amb.reporting.report import group_runs

    groups = group_runs(
        [
            {"dataset": "longmemeval", "variant": "oracle", "system": "mem0"},
            {"dataset": "longmemeval", "variant": "s", "system": "mem0"},
        ]
    )
    assert [g.plot_dir for g in groups] == [
        Path("plots/longmemeval/oracle"),
        Path("plots/longmemeval/s"),
    ]


def test_charted_score_is_judge_accuracy_alone():
    from amb.reporting.report import RunGroup

    group = RunGroup(key=(("dataset", "locomo"),), summaries=[])
    assert group.metrics() == (("accuracy", "judge_accuracy"),)


def test_group_summary_breaks_accuracy_down_by_category():
    from amb.contracts import Figure
    from amb.reporting.report import RunGroup
    from amb.reporting.run import ComparisonReport
    from amb.reporting.sections import GroupSummary

    run = {
        "run_id": "20260901T000000Z",
        "system": "alpha",
        "system_version": "1.0",
        "dataset": "locomo",
        "k": 10,
        "judge_accuracy": 0.54,
        "by_category": {"temporal": {"judge_accuracy": 0.2}},
    }
    section = GroupSummary(
        RunGroup(key=(("dataset", "locomo"),), summaries=[run]),
        ComparisonReport([run]),
        k=10,
    )
    assert section.categories is not None
    assert section.categories.path.name == "categories_k10.png"
    # declared and linked from the same chart
    assert section.categories in section.charts()
    assert any(
        isinstance(b, Figure) and b.path == section.categories.path
        for b in section.blocks()
    )
    header, rows = section.categories._category_table()
    assert header == ["system", "version", "temporal", "overall"]
    assert rows[0] == ["alpha", "1.0", "**0.200**", "**0.540**"]


def _attributed(system: str, k: int, found: float, when_found: float) -> dict:
    return {
        "run_id": f"20260901T000000Z-{system}",
        "system": system,
        "system_version": "1.0",
        "dataset": "locomo",
        "k": k,
        "judge_accuracy": 0.5,
        "attribution": {
            "correct": found * when_found,
            "agent_fault": found * (1 - when_found),
            "memory_fault": 1 - found,
            "lucky": 0.0,
            "found": found,
            "missed": 1 - found,
            "correct_when_found": when_found,
            "correct_when_missed": 0.0,
            "lift": found * when_found,
            "n": 100,
        },
        "memory_lift": found * when_found,
    }


def test_attribution_tables_mirror_each_other_in_one_row_order():
    from amb.reporting.run import ComparisonReport

    report = ComparisonReport(
        [
            _attributed("alpha", 10, found=0.6, when_found=0.9),
            _attributed("beta", 10, found=0.8, when_found=0.5),
        ]
    )
    header, rows = report.attribution_table(10, branch="found")
    assert header == [
        "system",
        "version",
        "correct with evidence",
        "wrong with evidence",
        "found",
    ]
    # most found first; the verdict columns are shares of the branch, not of all
    assert rows[0] == ["beta", "1.0", "0.500", "0.500", "0.800"]
    assert rows[1] == ["alpha", "1.0", "0.900", "0.100", "0.600"]
    header, rows = report.attribution_table(10, branch="missed")
    assert header == [
        "system",
        "version",
        "correct without evidence",
        "wrong without evidence",
        "missed",
    ]
    assert rows[0] == ["beta", "1.0", "0.000", "1.000", "0.200"]
    assert rows[1] == ["alpha", "1.0", "0.000", "1.000", "0.400"]


def test_attribution_table_refuses_an_unknown_branch():
    from amb.reporting.run import ComparisonReport

    report = ComparisonReport([_attributed("alpha", 10, 0.6, 0.9)])
    with pytest.raises(ValueError, match="found or missed"):
        report.attribution_table(10, branch="lost")


def test_attribution_table_leaves_out_runs_scored_before_the_metric():
    from amb.reporting.run import ComparisonReport

    older = {"run_id": "r", "system": "gamma", "dataset": "locomo", "k": 10}
    report = ComparisonReport([_attributed("alpha", 10, 0.6, 0.9), older])
    _, rows = report.attribution_table(10)
    assert [row[0] for row in rows] == ["alpha"]
    assert report.attribution_table(5) == ([], [])


def test_group_charts_show_both_error_tables_per_k():
    from amb.contracts import Heading, Table
    from amb.reporting.report import RunGroup
    from amb.reporting.run import ComparisonReport
    from amb.reporting.sections import GroupCharts

    summaries = [_attributed("alpha", 5, 0.6, 0.9), _attributed("alpha", 10, 0.7, 0.9)]
    comparison = ComparisonReport(summaries)
    section = GroupCharts(
        RunGroup(key=(("dataset", "locomo"),), summaries=summaries), comparison
    )
    blocks = section.blocks()
    headings = [b.text for b in blocks if isinstance(b, Heading)]
    assert "locomo: error analysis (k=5)" in headings
    assert "locomo: error analysis (k=10)" in headings
    tables = [b for b in blocks if isinstance(b, Table)]
    # found, missed, then their product at k=5; the same three at k=10
    assert [t.header[-1] for t in tables] == ["found", "missed", "lift"] * 2
    assert [t.rows[0][-1] for t in tables] == [
        "0.600",
        "0.400",
        "0.540",
        "0.700",
        "0.300",
        "0.630",
    ]


def test_lift_table_reads_as_found_times_gain():
    from amb.reporting.run import ComparisonReport

    report = ComparisonReport(
        [
            _attributed("alpha", 10, found=0.6, when_found=0.9),
            _attributed("beta", 10, found=0.8, when_found=0.5),
        ]
    )
    header, rows = report.lift_table(10)
    assert header == [
        "system",
        "version",
        "found",
        "correct with evidence",
        "correct without evidence",
        "gain",
        "lift",
    ]
    # most lift first, not most found
    assert rows[0] == ["alpha", "1.0", "0.600", "0.900", "0.000", "0.900", "0.540"]
    assert rows[1] == ["beta", "1.0", "0.800", "0.500", "0.000", "0.500", "0.400"]
    assert report.lift_table(5) == ([], [])


def test_summary_table_carries_the_lift_beside_accuracy():
    from amb.reporting.run import ComparisonReport

    header, rows = ComparisonReport([_attributed("alpha", 10, 0.6, 0.9)]).summary_table(
        10
    )
    assert header.index("lift") == header.index("accuracy") + 1
    assert rows[0][header.index("lift")] == "0.540"


def test_group_summary_breaks_the_lift_down_by_category():
    from amb.contracts import Figure
    from amb.reporting.report import RunGroup
    from amb.reporting.run import ComparisonReport
    from amb.reporting.sections import GroupSummary

    run = _attributed("alpha", 10, 0.6, 0.9)
    run["by_category"] = {"temporal": {"memory_lift": 0.2, "judge_accuracy": 0.5}}
    section = GroupSummary(
        RunGroup(key=(("dataset", "locomo"),), summaries=[run]),
        ComparisonReport([run]),
        k=10,
    )
    assert section.lifts is not None
    assert section.lifts.path.name == "lift_categories_k10.png"
    # declared and linked from the same chart
    assert section.lifts in section.charts()
    assert any(
        isinstance(b, Figure) and b.path == section.lifts.path for b in section.blocks()
    )
    header, rows = section.lifts._category_table()
    assert header == ["system", "version", "temporal", "overall"]
    assert rows[0] == ["alpha", "1.0", "**0.200**", "**0.540**"]
