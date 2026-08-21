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

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from amb.reporting import helpers, plots

if TYPE_CHECKING:
    from amb.reporting.report import BenchmarkReport


@dataclass
class Chart:
    """One figure: what it plots, from which runs, and where it lands.

    The unit both consumers share — `Section.blocks()` links exactly the
    charts `Section.charts()` declares, and `amb plot all` draws exactly
    those. `summaries` are its group's runs; a chart with a `k` narrows to
    that budget itself, so its identity travels with it.
    """

    kind: str  # bars | lines | scatter | table
    stem: str
    y: str
    out_dir: Path
    summaries: list[dict] = field(repr=False)
    alt: str = ""
    title: str = ""
    subtitle: str = ""
    x: str | None = None
    k: int | None = None
    better: str | None = None
    # the system whose score is the floor a real system must beat; drawn as
    # a reference line. Only meaningful on score-vs-cost scatters — a
    # baseline on cost-vs-cost would mark nothing.
    baseline_system: str | None = None

    @property
    def path(self) -> Path:
        """Where the image lands, relative to the working directory."""
        return self.out_dir / f"{self.stem}.png"

    def scoped(self) -> list[dict]:
        """The runs this chart plots: its group's, at its k when it names one."""
        if self.k is None:
            return self.summaries
        return [s for s in self.summaries if s.get("k") == self.k]

    def data(self) -> list:
        """The marks this chart would draw — empty when no run has the metric.

        Raises:
            ValueError: on a chart kind this module does not know.
        """
        scoped = self.scoped()
        match self.kind:
            case "bars":
                # scoped to one k, so each system contributes a single point
                return [
                    (line.label, line.ys[0])
                    for line in helpers.collect_series(scoped, self.y)
                ]
            case "lines":
                return helpers.collect_series(scoped, self.y)
            case "scatter":
                return helpers.collect_points(scoped, self.x or "", self.y)
            case "table":
                _, rows = self._table()
                return rows
            case _:
                raise ValueError(f"unknown chart kind {self.kind!r}")

    def _table(self) -> tuple[list[str], list[list[str]]]:
        """The summary table this chart renders — the README headline rows."""
        # imported here: run.py owns the table shape and imports nothing
        # from this module, but the reporting package initializes run
        # before chart, so a top-level import would be circular
        from amb.reporting.run import ComparisonReport

        return ComparisonReport(self.scoped()).summary_table(self.k or 10)

    def has_data(self) -> bool:
        """Whether any run in scope reports what this chart plots.

        Sections drop dataless charts before referencing them, so a
        document never links an image that was skipped.
        """
        return bool(self.data())

    def baseline(self) -> float | None:
        """The baseline system's score on this chart's metric, if it ran."""
        if self.baseline_system is None:
            return None
        for line in helpers.collect_series(self.scoped(), self.y):
            if line.label == self.baseline_system:
                return line.ys[0]
        return None

    def draw(self, dark: bool = False) -> Path | None:
        """Render the figure to `path`, or return None when it has no data.

        Raises:
            ValueError: on a chart kind this module does not know.
        """
        data = self.data()
        if not data:
            return None
        # dark renders land beside the light ones (`<stem>_dark.png`): the
        # README serves them through <picture> for dark-mode readers, and
        # the publish job regenerates both variants together
        output = self.out_dir / f"{self.stem}_dark.png" if dark else self.path
        match self.kind:
            case "bars":
                return plots.bars(
                    data,
                    x_label=self.y,
                    output=output,
                    title=self.title,
                    subtitle=self.subtitle,
                    dark=dark,
                )
            case "lines":
                return plots.lines(
                    data,
                    x_label="k (hits requested per query)",
                    y_label=self.y,
                    output=output,
                    title=self.title,
                    subtitle=self.subtitle,
                    dark=dark,
                )
            case "scatter":
                floor = self.baseline()
                subtitle = self.subtitle + (
                    " · line marks the naive (BM25) floor" if floor is not None else ""
                )
                return plots.scatter(
                    data,
                    x_label=self.x or "",
                    y_label=self.y,
                    output=output,
                    title=self.title,
                    subtitle=subtitle,
                    better=self.better,
                    baseline=floor,
                    dark=dark,
                )
            case "table":
                header, rows = self._table()
                return plots.table(
                    header,
                    rows,
                    output=output,
                    title=self.title,
                    subtitle=self.subtitle,
                    dark=dark,
                )
            case _:
                raise ValueError(f"unknown chart kind {self.kind!r}")


def plan_charts(
    reports: "Sequence[BenchmarkReport]", out: Path | None = None
) -> list[Chart]:
    """Every chart the reports show, deduplicated, optionally redirected.

    `out` overrides the destination directory — a chart-only convenience
    (`amb plot all --out`), never used when documents are written, since
    their figure links follow the group's own `plots/` layout.
    """
    seen: dict[str, Chart] = {}
    for report in reports:
        for chart in report.charts():
            chart = chart if out is None else replace(chart, out_dir=out)
            seen.setdefault(chart.path.as_posix(), chart)
    return list(seen.values())
