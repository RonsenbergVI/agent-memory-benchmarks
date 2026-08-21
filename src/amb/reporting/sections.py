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

from dataclasses import dataclass, field

from amb.base.reporting import Section
from amb.constants import LATENCY, SUMMARY_ORDER, TOKENS
from amb.contracts import Block, Figure, Heading, Paragraph, Rule, Table
from amb.reporting.chart import Chart
from amb.reporting.helpers import headline, pretty
from amb.reporting.report import RunGroup, precision_pinned
from amb.reporting.run import ComparisonReport


@dataclass
class Prose(Section):
    """Static blocks — a title, an introduction, a closing note."""

    content: list[Block]

    def charts(self) -> list[Chart]:
        """No charts: prose shows none."""
        return []

    def blocks(self) -> list[Block]:
        """The blocks as given."""
        return list(self.content)


@dataclass
class ComparisonTable(Section):
    """The full cross-system table: every run, every identity column."""

    comparison: ComparisonReport
    heading: str | None = None
    level: int = 2
    intro: str | None = None

    def charts(self) -> list[Chart]:
        """No charts: the table is the whole section."""
        return []

    def blocks(self) -> list[Block]:
        """Optional heading and intro, then the table itself."""
        blocks: list[Block] = []
        if self.heading:
            blocks.append(Heading(level=self.level, text=self.heading))
        if self.intro:
            blocks.append(Paragraph(text=self.intro))
        header, rows = self.comparison.table()
        blocks.append(Table(header=header, rows=rows))
        return blocks


@dataclass
class GroupCharts(Section):
    """One group's chart sets — the single-k detail, one set per k it ran.

    Every k the group's runs used gets its own set, so the document covers
    the whole sweep instead of one budget somebody remembered to pass on
    the command line. Charts no run has data for are dropped up front,
    never linked and never drawn.
    """

    group: RunGroup
    level: int = 3
    metric_filter: tuple[str, ...] = ()
    # the group's k-sweep lines, shown before the per-k sets. The same
    # figures GroupSummary links — planned with identical paths, so the
    # reports' chart sets deduplicate to one drawing
    sweeps: list[Chart] = field(default_factory=list, init=False, repr=False)
    # per k: its bar charts and its trade-off scatters, both non-empty only
    sets: list[tuple[int, list[Chart], list[Chart]]] = field(
        default_factory=list, init=False, repr=False
    )
    # charts the metric set calls for that no run has data for: reported
    # rather than silently absent
    skipped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Plan the chart sets, keeping only what the runs have data for."""
        metrics = self.group.metrics(self.metric_filter)
        planned = [
            Chart(
                kind="lines",
                stem=f"k_{stem}",
                y=metric,
                out_dir=self.group.plot_dir,
                summaries=self.group.summaries,
                alt=f"Retrieval {pretty(stem)} vs k",
                title=headline(f"{pretty(metric)} vs k"),
                subtitle=f"{self.group.label} · {self.group.mode} · "
                "newest run per system and k",
            )
            for stem, metric in sorted(
                metrics, key=lambda pair: SUMMARY_ORDER.index(pair[0])
            )
        ]
        self.sweeps = [c for c in planned if c.has_data()]
        self.skipped += len(planned) - len(self.sweeps)
        for k in self.group.ks():
            bars = [self._bars(metric, k) for _, metric in metrics]
            scatters = [
                self._scatter(x, y, stem, k)
                for x, y, stem in (
                    *((TOKENS, metric, f"tokens_{stem}") for stem, metric in metrics),
                    *((LATENCY, metric, f"latency_{stem}") for stem, metric in metrics),
                    (TOKENS, LATENCY, "tokens_latency"),
                )
            ]
            planned = len(bars) + len(scatters)
            bars = [c for c in bars if c.has_data()]
            scatters = [c for c in scatters if c.has_data()]
            self.skipped += planned - len(bars) - len(scatters)
            if bars or scatters:
                self.sets.append((k, bars, scatters))

    def _bars(self, metric: str, k: int) -> Chart:
        """One metric's system comparison at `k`."""
        stem = metric.removeprefix("retrieval_")
        return Chart(
            kind="bars",
            stem=f"session_{stem}_k{k}",
            y=metric,
            k=k,
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"Session-level {pretty(stem)} by system",
            title=headline(f"{pretty(metric)} by system"),
            subtitle=f"{self.group.label} · {self.group.mode} · k={k} · "
            "newest run per system",
        )

    def _scatter(self, x: str, y: str, stem: str, k: int) -> Chart:
        """One quality-against-cost trade-off at `k`."""
        return Chart(
            kind="scatter",
            stem=f"{stem}_k{k}",
            x=x,
            y=y,
            k=k,
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"{headline(pretty(y))} vs {pretty(x)}",
            title=headline(f"{pretty(y)} vs {pretty(x)}"),
            subtitle=f"{self.group.label} · {self.group.mode} · k={k}",
            better=(
                r"$\downarrow$ faster, $\leftarrow$ cheaper"
                if stem == "tokens_latency"
                else r"$\uparrow$ better, $\leftarrow$ faster"
                if stem.startswith("latency")
                else r"$\uparrow$ better, $\leftarrow$ cheaper"
            ),
            # a cost-vs-cost scatter names no score, so it has no floor
            baseline_system=None if stem == "tokens_latency" else "naive",
        )

    def charts(self) -> list[Chart]:
        """The k-sweep lines, then every chart in every k's set."""
        per_k = [c for _, bars, scatters in self.sets for c in (*bars, *scatters)]
        return [*self.sweeps, *per_k]

    def blocks(self) -> list[Block]:
        """The group's heading, why its metric set is what it is, then each k."""
        blocks: list[Block] = [Heading(level=self.level, text=self.group.label)]
        if not self.metric_filter and precision_pinned(self.group.summaries):
            blocks.append(
                Paragraph(
                    text="`retrieval_precision` reads 1.000 for every system at "
                    "every k here by construction — this variant's haystack ships "
                    "only the evidence sessions, so no hit can be a false "
                    "positive. Only the recall charts are written: precision "
                    "carries no information, and F1 collapses to a pure function "
                    "of recall once precision is pinned at 1 (F1 = 2·1·R/(1+R))."
                )
            )
        if self.sweeps:
            blocks.append(
                Heading(
                    level=self.level + 1, text=f"{self.group.label}: retrieval vs k"
                )
            )
            blocks.append(
                Paragraph(
                    text="One line per system across the k sweep, newest run "
                    "per system and k."
                )
            )
            blocks += [Figure(alt=c.alt, path=c.path) for c in self.sweeps]
        for index, (k, bars, scatters) in enumerate(self.sets):
            if index or self.sweeps:
                blocks.append(Rule())
            if bars:
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: session-level comparison (k={k})",
                    )
                )
                blocks.append(Paragraph(text="One bar per system, newest run."))
                blocks += [Figure(alt=c.alt, path=c.path) for c in bars]
            if scatters:
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: cross-metric trade-offs (k={k})",
                    )
                )
                blocks.append(
                    Paragraph(
                        text="Retrieval quality against memory tokens spent and "
                        "search latency."
                    )
                )
                blocks += [Figure(alt=c.alt, path=c.path) for c in scatters]
        return blocks


@dataclass
class GroupSummary(Section):
    """One group's headline: a row per system at one k, plus its k sweeps.

    The overview half of the pair — `GroupCharts` is the detail. The sweep
    lines belong here because they answer "how does this system behave as
    the budget grows" in one chart per metric.
    """

    group: RunGroup
    comparison: ComparisonReport
    k: int
    level: int = 3
    metric_filter: tuple[str, ...] = ()
    lines: list[Chart] = field(default_factory=list, init=False, repr=False)
    # the compact table at `k`, rendered as a figure: the README embeds an
    # image from plots/, where CI blocks hand edits, instead of markdown
    # numbers anyone could quietly change (the exact numbers stay in
    # RESULTS.md, which is guarded the same way)
    summary: Chart | None = field(default=None, init=False, repr=False)
    skipped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Plan the table and sweep charts, keeping what runs have data for."""
        table = Chart(
            kind="table",
            stem=f"summary_k{self.k}",
            y="retrieval_f1",
            k=self.k,
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"Cross-system summary at k={self.k}",
            title=headline(f"summary at k={self.k}"),
            subtitle=f"{self.group.label} · {self.group.mode} · "
            f"newest run per system{self._run_stamp()}",
        )
        self.summary = table if table.has_data() else None
        planned = [
            Chart(
                kind="lines",
                stem=f"k_{stem}",
                y=metric,
                out_dir=self.group.plot_dir,
                summaries=self.group.summaries,
                alt=f"Retrieval {pretty(stem)} vs k",
                title=headline(f"{pretty(metric)} vs k"),
                subtitle=f"{self.group.label} · {self.group.mode} · "
                "newest run per system and k",
            )
            for stem, metric in sorted(
                self.group.metrics(self.metric_filter),
                key=lambda pair: SUMMARY_ORDER.index(pair[0]),
            )
        ]
        self.lines = [c for c in planned if c.has_data()]
        self.skipped = len(planned) - len(self.lines) + (0 if self.summary else 1)

    def _run_stamp(self) -> str:
        """The newest run's date, carried inside the image where it can't drift."""
        newest = max((s.get("run_id") or "" for s in self.group.summaries), default="")
        if len(newest) < 8 or not newest[:8].isdigit():
            return ""
        return f" · run {newest[:4]}-{newest[4:6]}-{newest[6:8]}"

    def charts(self) -> list[Chart]:
        """The group's summary table figure and its k-sweep charts."""
        return ([self.summary] if self.summary else []) + list(self.lines)

    def blocks(self) -> list[Block]:
        """Heading, the table figure at `k`, then the sweep charts."""
        blocks: list[Block] = [Heading(level=self.level, text=self.group.label)]
        if self.summary:
            blocks.append(Figure(alt=self.summary.alt, path=self.summary.path))
        else:
            # no run at this k: fall back to the markdown table rather than
            # linking an image that was never drawn
            header, rows = self.comparison.summary_table(
                self.k, dataset=self.group.dataset, variant=self.group.variant or None
            )
            blocks.append(Table(header=header, rows=rows))
        blocks += [Figure(alt=c.alt, path=c.path) for c in self.lines]
        return blocks
