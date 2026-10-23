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
from amb.constants import (
    ATTRIBUTION_FOUND,
    ATTRIBUTION_INTRO,
    ATTRIBUTION_MISSED,
    HEADLINE,
    LATENCY,
    LIFT,
    LIFT_INTRO,
    TOKENS,
)
from amb.contracts import Block, Figure, Heading, Paragraph, Rule, Table
from amb.reporting.chart import Chart
from amb.reporting.helpers import headline, pretty, slug
from amb.reporting.report import RunGroup
from amb.reporting.run import ComparisonReport, run_date


def freshness(summaries: list[dict]) -> str:
    """`· run YYYY-MM-DD` of the newest run in scope, or "" without one.

    Carried in every chart subtitle so a saved image dates itself.
    """
    stamp = run_date(max((s.get("run_id") or "" for s in summaries), default=""))
    return f" · run {stamp}" if stamp else ""


def spend_sweep(group: RunGroup, stem: str, metric: str) -> Chart:
    """One metric against the hits each k actually returned, a line per system.

    k is the budget the harness grants; hits per query is what a system
    spends of it. At a matched spend, returning fewer hits than allowed
    cannot pass for ranking better.
    """
    return Chart(
        kind="lines",
        stem=f"hits_{stem}",
        x="hits_per_query",
        x_label="hits returned per query (mean)",
        y=metric,
        out_dir=group.plot_dir,
        summaries=group.summaries,
        alt=f"{headline(pretty(metric))} vs hits returned per query",
        title=headline(f"{pretty(metric)} vs hits returned"),
        subtitle=f"{group.label} · a point per k, "
        f"newest run per system and k{freshness(group.summaries)}",
    )


def category_breakdown(group: RunGroup, k: int) -> Chart:
    """Judge accuracy per question category at `k`, a system per row.

    One definition for both reports, since they share its image path.
    """
    return Chart(
        kind="categories",
        stem=f"categories_k{k}",
        y=HEADLINE,
        k=k,
        out_dir=group.plot_dir,
        summaries=group.summaries,
        alt=f"Judge accuracy by question category at k={k}",
        title=headline(f"accuracy by question category at k={k}"),
        subtitle=f"{group.label} · newest run per system"
        f"{freshness([s for s in group.summaries if s.get('k') == k])}",
    )


def lift_breakdown(group: RunGroup, k: int) -> Chart:
    """Memory lift per question category at `k`, a system per row."""
    return Chart(
        kind="categories",
        stem=f"lift_categories_k{k}",
        y=LIFT,
        k=k,
        out_dir=group.plot_dir,
        summaries=group.summaries,
        alt=f"Memory lift by question category at k={k}",
        title=headline(f"memory lift by question category at k={k}"),
        subtitle=f"{group.label} · newest run per system"
        f"{freshness([s for s in group.summaries if s.get('k') == k])}",
    )


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

    Every k the runs used gets its own set, covering the whole sweep.
    Charts no run has data for are dropped up front, never linked or drawn.
    """

    group: RunGroup
    comparison: ComparisonReport
    level: int = 3
    metric_filter: tuple[str, ...] = ()
    # k-sweep lines; same figures (and paths) GroupSummary links, so the
    # reports' chart sets deduplicate to one drawing
    sweeps: list[Chart] = field(default_factory=list, init=False, repr=False)
    # the same sweeps against hits returned, shared with GroupSummary likewise
    spends: list[Chart] = field(default_factory=list, init=False, repr=False)
    # per k: its bar charts and its trade-off scatters
    sets: list[tuple[int, list[Chart], list[Chart]]] = field(
        default_factory=list, init=False, repr=False
    )
    # one accuracy sweep per question category: the breakdown table over k
    category_sweeps: list[Chart] = field(default_factory=list, init=False, repr=False)
    # per k: the question-category breakdown, when the runs label questions
    breakdowns: dict[int, Chart] = field(default_factory=dict, init=False, repr=False)
    # planned charts with no data: reported rather than silently absent
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
                alt=f"{headline(pretty(metric))} vs k",
                title=headline(f"{pretty(metric)} vs k"),
                subtitle=f"{self.group.label} · "
                f"newest run per system and k{freshness(self.group.summaries)}",
            )
            for stem, metric in metrics
        ]
        self.sweeps = [c for c in planned if c.has_data()]
        self.skipped += len(planned) - len(self.sweeps)
        planned = [spend_sweep(self.group, stem, metric) for stem, metric in metrics]
        self.spends = [c for c in planned if c.has_data()]
        self.skipped += len(planned) - len(self.spends)
        planned = [self._category_sweep(c) for c in self.group.categories()]
        self.category_sweeps = [c for c in planned if c.has_data()]
        self.skipped += len(planned) - len(self.category_sweeps)
        for k in self.group.ks():
            bars = [self._bars(stem, metric, k) for stem, metric in metrics]
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
            breakdown = category_breakdown(self.group, k)
            if breakdown.has_data():
                self.breakdowns[k] = breakdown
            else:
                self.skipped += 1
            if bars or scatters:
                self.sets.append((k, bars, scatters))

    def _at_k(self, k: int) -> list[dict]:
        """The group's runs at one k — what that k's charts actually show."""
        return [s for s in self.group.summaries if s.get("k") == k]

    def _category_sweep(self, category: str) -> Chart:
        """One category's judge accuracy across k, a line per system.

        The breakdown table says where a system stands at one k; this says
        whether a bigger budget is what it was missing.
        """
        return Chart(
            kind="lines",
            stem=f"k_accuracy_{slug(category)}",
            y=f"by_category.{category}.{HEADLINE}",
            y_label=f"{category} accuracy",
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"Judge accuracy vs k for {category} questions",
            title=headline(f"{category} accuracy vs k"),
            subtitle=f"{self.group.label} · "
            f"newest run per system and k{freshness(self.group.summaries)}",
        )

    def _bars(self, stem: str, metric: str, k: int) -> Chart:
        """One score's system comparison at `k`."""
        return Chart(
            kind="bars",
            stem=f"{stem}_k{k}",
            y=metric,
            k=k,
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"{headline(pretty(metric))} by system",
            title=headline(f"{pretty(metric)} by system"),
            subtitle=f"{self.group.label} · k={k} · "
            f"newest run per system{freshness(self._at_k(k))}",
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
            subtitle=f"{self.group.label} · k={k}{freshness(self._at_k(k))}",
            better=(
                r"$\downarrow$ faster, $\leftarrow$ cheaper"
                if stem == "tokens_latency"
                else r"$\uparrow$ better, $\leftarrow$ faster"
                if stem.startswith("latency")
                else r"$\uparrow$ better, $\leftarrow$ cheaper"
            ),
            # cost-vs-cost names no score, so no floor
            baseline_system=None if stem == "tokens_latency" else "naive",
        )

    def charts(self) -> list[Chart]:
        """The k-sweep lines, then every chart in every k's set."""
        per_k = [c for _, bars, scatters in self.sets for c in (*bars, *scatters)]
        return [
            *self.sweeps,
            *self.spends,
            *self.category_sweeps,
            *per_k,
            *self.breakdowns.values(),
        ]

    def blocks(self) -> list[Block]:
        """The group's heading, why its metric set is what it is, then each k."""
        blocks: list[Block] = [Heading(level=self.level, text=self.group.label)]
        if self.sweeps:
            blocks.append(
                Heading(level=self.level + 1, text=f"{self.group.label}: scores vs k")
            )
            blocks.append(
                Paragraph(
                    text="One line per system across the k sweep, newest run "
                    "per system and k."
                )
            )
            blocks += [Figure(alt=c.alt, path=c.path) for c in self.sweeps]
        if self.spends:
            blocks.append(
                Heading(
                    level=self.level + 1,
                    text=f"{self.group.label}: scores vs hits returned",
                )
            )
            blocks.append(
                Paragraph(
                    text="The same sweep, placed at the hits each k actually "
                    "returned: k is the budget granted, hits per query what a "
                    "system spent of it. Compare systems at a matched spend, "
                    "where returning fewer hits than allowed cannot pass for "
                    "ranking better."
                )
            )
            blocks += [Figure(alt=c.alt, path=c.path) for c in self.spends]
        if self.category_sweeps:
            blocks.append(
                Heading(
                    level=self.level + 1,
                    text=f"{self.group.label}: accuracy vs k by question category",
                )
            )
            blocks.append(
                Paragraph(
                    text="The breakdown table at one k, swept: whether a system "
                    "is weak on a question type, or simply needed a bigger budget."
                )
            )
            blocks += [Figure(alt=c.alt, path=c.path) for c in self.category_sweeps]
        for index, (k, bars, scatters) in enumerate(self.sets):
            if index or self.sweeps:
                blocks.append(Rule())
            if k in self.breakdowns:
                chart = self.breakdowns[k]
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: by question category (k={k})",
                    )
                )
                blocks.append(
                    Paragraph(
                        text="Where each system answers correctly, not just how "
                        "often overall. The best accuracy in each column is bold."
                    )
                )
                blocks.append(Figure(alt=chart.alt, path=chart.path))
            found, missed = (
                self.comparison.attribution_table(
                    k,
                    dataset=self.group.dataset,
                    variant=self.group.variant or None,
                    branch=branch,
                )
                for branch in ("found", "missed")
            )
            if found[1]:
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: error analysis (k={k})",
                    )
                )
                blocks.append(Paragraph(text=ATTRIBUTION_INTRO))
                blocks.append(Paragraph(text=ATTRIBUTION_FOUND))
                blocks.append(Table(header=found[0], rows=found[1]))
                blocks.append(Paragraph(text=ATTRIBUTION_MISSED))
                blocks.append(Table(header=missed[0], rows=missed[1]))
            header, rows = self.comparison.lift_table(
                k, dataset=self.group.dataset, variant=self.group.variant or None
            )
            if rows:
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: memory lift (k={k})",
                    )
                )
                blocks.append(Paragraph(text=LIFT_INTRO))
                blocks.append(Table(header=header, rows=rows))
            if bars:
                blocks.append(
                    Heading(
                        level=self.level + 1,
                        text=f"{self.group.label}: system comparison (k={k})",
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
                        text="Answer accuracy against memory tokens spent and "
                        "search latency."
                    )
                )
                blocks += [Figure(alt=c.alt, path=c.path) for c in scatters]
        return blocks


@dataclass
class GroupSummary(Section):
    """One group's headline: a row per system at one k, plus its k sweeps.

    The overview half of the pair — `GroupCharts` is the detail.
    """

    group: RunGroup
    comparison: ComparisonReport
    k: int
    level: int = 3
    metric_filter: tuple[str, ...] = ()
    lines: list[Chart] = field(default_factory=list, init=False, repr=False)
    spends: list[Chart] = field(default_factory=list, init=False, repr=False)
    # the compact table at `k` rendered as a figure: the README embeds an image
    # from plots/, where CI blocks hand edits, not editable markdown numbers
    summary: Chart | None = field(default=None, init=False, repr=False)
    # the same rows broken out by question category, when the runs label them
    categories: Chart | None = field(default=None, init=False, repr=False)
    # the memory's contribution per question category, likewise
    lifts: Chart | None = field(default=None, init=False, repr=False)
    skipped: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Plan the table and sweep charts, keeping what runs have data for."""
        table = Chart(
            kind="table",
            stem=f"summary_k{self.k}",
            y=HEADLINE,
            k=self.k,
            out_dir=self.group.plot_dir,
            summaries=self.group.summaries,
            alt=f"Cross-system summary at k={self.k}",
            title=headline(f"summary at k={self.k}"),
            subtitle=f"{self.group.label} · "
            f"newest run per system{freshness(self.group.summaries)}",
        )
        self.summary = table if table.has_data() else None
        breakdown = category_breakdown(self.group, self.k)
        # a dataset whose runs carry no question labels simply has no table
        self.categories = breakdown if breakdown.has_data() else None
        lifts = lift_breakdown(self.group, self.k)
        self.lifts = lifts if lifts.has_data() else None
        planned = [
            Chart(
                kind="lines",
                stem=f"k_{stem}",
                y=metric,
                out_dir=self.group.plot_dir,
                summaries=self.group.summaries,
                alt=f"{headline(pretty(metric))} vs k",
                title=headline(f"{pretty(metric)} vs k"),
                subtitle=f"{self.group.label} · "
                f"newest run per system and k{freshness(self.group.summaries)}",
            )
            for stem, metric in self.group.metrics(self.metric_filter)
        ]
        self.lines = [c for c in planned if c.has_data()]
        self.skipped = len(planned) - len(self.lines) + (0 if self.summary else 1)
        planned = [
            spend_sweep(self.group, stem, metric)
            for stem, metric in self.group.metrics(self.metric_filter)
        ]
        self.spends = [c for c in planned if c.has_data()]
        self.skipped += len(planned) - len(self.spends)

    def charts(self) -> list[Chart]:
        """The group's summary tables, its k sweeps, and their spend twins."""
        tables = [c for c in (self.summary, self.categories, self.lifts) if c]
        return tables + list(self.lines) + list(self.spends)

    def blocks(self) -> list[Block]:
        """Heading, the table figure at `k`, then the sweep charts."""
        blocks: list[Block] = [Heading(level=self.level, text=self.group.label)]
        if self.summary:
            blocks.append(Figure(alt=self.summary.alt, path=self.summary.path))
        else:
            # no run at this k: markdown table, not a link to an undrawn image
            header, rows = self.comparison.summary_table(
                self.k, dataset=self.group.dataset, variant=self.group.variant or None
            )
            blocks.append(Table(header=header, rows=rows))
        if self.categories:
            blocks.append(Figure(alt=self.categories.alt, path=self.categories.path))
        if self.lifts:
            blocks.append(Figure(alt=self.lifts.alt, path=self.lifts.path))
        blocks += [Figure(alt=c.alt, path=c.path) for c in self.lines]
        blocks += [Figure(alt=c.alt, path=c.path) for c in self.spends]
        return blocks
