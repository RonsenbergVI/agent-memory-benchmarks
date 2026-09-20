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


def _categorised(system: str, version: str, k: int, f1: float, cats: dict) -> dict:
    return {
        "run_id": f"20260901T000000Z-{system}",
        "system": system,
        "system_version": version,
        "dataset": "locomo",
        "k": k,
        "retrieval_f1": f1,
        "by_category": {c: {"retrieval_f1": v} for c, v in cats.items()},
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
