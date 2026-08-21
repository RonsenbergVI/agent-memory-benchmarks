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

"""The table figure: the README's guarded replacement for markdown numbers."""

from pathlib import Path

from amb.reporting.chart import Chart


def _summary(system: str, f1: float) -> dict:
    return {
        "run_id": "20260820T000000Z",
        "system": system,
        "system_version": "1.0",
        "dataset": "locomo",
        "mode": "direct",
        "k": 10,
        "retrieval_precision": 0.4,
        "retrieval_recall": 0.9,
        "retrieval_f1": f1,
        "memory_tokens_total": 1000.0,
        "search_latency": {"p50_s": 0.1},
    }


def _table_chart(tmp_path: Path, summaries: list[dict], k: int = 10) -> Chart:
    return Chart(
        kind="table",
        stem=f"summary_k{k}",
        y="retrieval_f1",
        k=k,
        out_dir=tmp_path,
        summaries=summaries,
        title=f"Summary at k={k}",
        subtitle="locomo · direct",
    )


def test_table_chart_ranks_and_renders(tmp_path: Path) -> None:
    chart = _table_chart(tmp_path, [_summary("aaa", 0.5), _summary("bbb", 0.6)])
    assert chart.has_data()
    # rows come ranked best F1 first, so the image argues the same order
    # as the markdown table it replaces
    assert [row[0] for row in chart.data()] == ["bbb", "aaa"]
    path = chart.draw()
    assert path == tmp_path / "summary_k10.png"
    assert path is not None
    assert path.stat().st_size > 0
    # the dark render lands beside the light one, never over it — the
    # README serves the pair through <picture>
    dark = chart.draw(dark=True)
    assert dark == tmp_path / "summary_k10_dark.png"
    assert dark is not None
    assert dark.stat().st_size > 0


def test_table_chart_without_runs_at_k_has_no_data(tmp_path: Path) -> None:
    # scoped to a k nobody ran: the chart reports no data, so sections
    # drop it instead of linking an image that was never drawn
    chart = _table_chart(tmp_path, [_summary("aaa", 0.5)], k=3)
    assert not chart.has_data()
    assert chart.draw() is None
