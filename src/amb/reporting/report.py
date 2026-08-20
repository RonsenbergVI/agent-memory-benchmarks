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

"""Generated documents: reports, their sections, and the charts they show.

RESULTS.md and the README's results section are *outputs of this module*,
never hand-edited files that happen to agree with `plots/`. One rule makes
that safe: **a section declares its charts, and renders its figures from
that same list.** Not two lists kept in sync — one list with two consumers,
so a document cannot link a chart nobody drew, and `amb plot all` cannot
draw a set no document shows.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from amb.base.reporting import Report, Section
from amb.constants import (
    DEFAULT_PLOT_DIR,
    GROUP_BY,
    GUARD,
    RESULTS_INTRO,
    RESULTS_OUTRO,
    RESULTS_PLOTS_INTRO,
    RESULTS_TABLE_INTRO,
    RETRIEVAL_METRICS,
    SUMMARY_MARKS,
)
from amb.contracts import Block, Heading, Paragraph
from amb.reporting.chart import Chart
from amb.reporting.helpers import slug
from amb.reporting.renderers import get_renderer
from amb.reporting.run import ComparisonReport


def splice(path: Path, marks: tuple[str, str], block: str, template: str) -> None:
    """Replace the marked region of `path`, creating it when missing.

    With both markers present only the text between them changes — the
    surrounding prose is never touched. Otherwise `template` (holding
    ``{block}``) writes the region fresh: appended when the file exists, as
    the whole file when it does not.
    """
    start, end = marks
    body = f"{start}\n{block}\n{end}"
    if path.exists():
        text = path.read_text()
        if start in text and end in text:
            before, _, rest = text.partition(start)
            _, _, after = rest.partition(end)
            path.write_text(before + body + after)
            return
        path.write_text(text.rstrip() + "\n\n" + template.format(block=body))
        return
    path.write_text(template.format(block=body))


def precision_pinned(summaries: list[dict]) -> bool:
    """Whether every summary in scope reports precision exactly 1.0.

    True on dataset variants whose haystack ships only evidence (e.g.
    LongMemEval's oracle) — no hit can ever be a false positive there, so
    precision carries no information and F1 (= 2R/(1+R) when precision is
    pinned at 1) is a pure function of recall. Both drop out of the chart
    set in that case, and the section says why.
    """
    scores = [s["retrieval_precision"] for s in summaries if "retrieval_precision" in s]
    return bool(scores) and all(p == 1.0 for p in scores)


@dataclass
class RunGroup:
    """The runs of one (dataset, variant, ...) — what a section covers.

    `key` is the group_by fields and their values, in order: it names the
    group, titles its section, and nests its chart directory.
    """

    key: tuple[tuple[str, str], ...]
    summaries: list[dict] = field(repr=False)

    @property
    def dataset(self) -> str:
        """The dataset these runs measured."""
        return dict(self.key).get("dataset", "")

    @property
    def variant(self) -> str:
        """The dataset variant, or "" when the dataset has none."""
        return dict(self.key).get("variant", "")

    @property
    def qualifiers(self) -> list[str]:
        """The group's identity beyond its dataset, e.g. ["oracle"]."""
        return [value for field_, value in self.key if field_ != "dataset" and value]

    @property
    def label(self) -> str:
        """How the group is named in a heading: `longmemeval (oracle)`."""
        rest = ", ".join(self.qualifiers)
        return f"{self.dataset} ({rest})" if rest else self.dataset

    @property
    def plot_dir(self) -> Path:
        """Where this group's charts live: `plots/<dataset>[/<qualifier>...]`.

        Chart filenames are not namespaced, so two groups sharing one
        directory would overwrite each other; the key nests them apart.
        """
        parts = [slug(value) for _, value in self.key if value]
        return DEFAULT_PLOT_DIR.joinpath(*parts)

    @property
    def mode(self) -> str:
        """The experiment kind these runs share (guarded to be uniform)."""
        modes = {str(s.get("mode", "")) for s in self.summaries}
        return modes.pop() if len(modes) == 1 else ""

    def ks(self) -> list[int]:
        """Every retrieval budget these runs used, ascending."""
        return sorted({s["k"] for s in self.summaries if isinstance(s.get("k"), int)})

    def metrics(self, only: Sequence[str] = ()) -> tuple[tuple[str, str], ...]:
        """The retrieval metrics worth charting for this group.

        `only` (filename stems) overrides the choice outright. Otherwise
        every metric is charted, except where precision is structurally
        pinned at 1.0 and only recall carries information.
        """
        selected = tuple(only) or (
            ("recall",)
            if precision_pinned(self.summaries)
            else tuple(stem for stem, _ in RETRIEVAL_METRICS)
        )
        return tuple((stem, key) for stem, key in RETRIEVAL_METRICS if stem in selected)


def group_runs(
    summaries: list[dict],
    group_by: Sequence[str] = GROUP_BY,
    guard: Sequence[str] = GUARD,
) -> list[RunGroup]:
    """Split run summaries into the groups sections are built from.

    Groups come back in a stable order, with unqualified values (a dataset
    with no variant) before qualified ones.

    Raises:
        ValueError: when a group's runs disagree on a `guard` field —
            blending them into one chart would compare different
            experiments as if they were one.
    """
    grouped: dict[tuple[tuple[str, str], ...], list[dict]] = {}
    for summary in summaries:
        key = tuple(
            (field_, ComparisonReport.identity_label(summary, field_))
            for field_ in group_by
        )
        grouped.setdefault(key, []).append(summary)

    groups = []
    for key in sorted(grouped, key=lambda k: [(v == "", v) for _, v in k]):
        members = grouped[key]
        for field_ in guard:
            values = {ComparisonReport.identity_label(s, field_) for s in members}
            if len(values) > 1:
                name = ", ".join(f"{k}={v}" for k, v in key if v)
                raise ValueError(
                    f"{name} spans several {field_} values "
                    f"({', '.join(sorted(values))}); filter to one, or add "
                    f"{field_!r} to group_by so each gets its own section"
                )
        groups.append(RunGroup(key=key, summaries=members))
    return groups


class BenchmarkReport(Report):
    """One generated document: its sections, its file, how it is written.

    `marks` decides the write mode. Without them the file is generated
    whole — everything it says comes from the runs, so there is nothing to
    preserve. With them only the marked region is rewritten, for a file
    like README.md whose hand-written prose surrounds the results.
    """

    def __init__(
        self,
        name: str,
        path: Path,
        sections: list[Section],
        marks: tuple[str, str] | None = None,
        template: str = "{block}\n",
        groups: list[RunGroup] | None = None,
    ) -> None:
        """Bind the document's sections to the file they are written into."""
        self.name = name
        self.path = path
        self.sections = sections
        self.marks = marks
        self.template = template
        # the run groups this report covers, one section each — kept for the
        # CLI's "wrote N section(s)" line
        self.groups = groups or []

    @property
    def skipped(self) -> int:
        """Charts its sections planned but dropped for want of data."""
        return sum(getattr(section, "skipped", 0) for section in self.sections)

    def charts(self) -> list[Chart]:
        """Every chart this report shows, deduplicated by output path.

        Reports overlap (the summary's sweep lines also appear elsewhere),
        so the same figure is planned once and drawn once.
        """
        seen: dict[str, Chart] = {}
        for section in self.sections:
            for chart in section.charts():
                seen.setdefault(chart.path.as_posix(), chart)
        return list(seen.values())

    def blocks(self) -> list[Block]:
        """Every section's blocks, in order."""
        return [block for section in self.sections for block in section.blocks()]

    def render(self, fmt: str = "markdown") -> str:
        """Render the whole document in the named format."""
        return get_renderer(fmt).render(self.blocks())

    def to_markdown(self) -> str:
        """Render the whole document as markdown."""
        return self.render("markdown")

    def write(self, fmt: str = "markdown") -> Path:
        """Write the document to `path` and return it."""
        text = self.render(fmt)
        if self.marks is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text)
        else:
            splice(self.path, self.marks, text.strip(), self.template)
        return self.path


def results_report(
    comparison: ComparisonReport,
    groups: list[RunGroup],
    *,
    k: int = 10,
    stamp: str = "",
    path: Path = Path("RESULTS.md"),
    metric_filter: tuple[str, ...] = (),
) -> BenchmarkReport:
    """Build RESULTS.md: the full table, then every group's chart detail.

    Written whole rather than spliced — nothing in it is hand-authored, so
    there is no prose to preserve. `k` is unused here (the file covers
    every k the runs used); it is in the signature so every report builder
    is callable the same way.
    """
    # imported here: sections use RunGroup/precision_pinned from this
    # module, so a top-level import would be circular
    from amb.reporting.sections import ComparisonTable, GroupCharts, Prose

    header: list[Block] = [
        Heading(level=1, text="Agent Memory Benchmark — Results"),
        Paragraph(text=RESULTS_INTRO),
    ]
    if stamp:
        header.append(Paragraph(text=stamp))
    sections: list[Section] = [
        Prose(header),
        ComparisonTable(comparison, heading="Every run", intro=RESULTS_TABLE_INTRO),
        Prose([Heading(level=2, text="Plots"), Paragraph(text=RESULTS_PLOTS_INTRO)]),
    ]
    sections += [
        GroupCharts(group, level=3, metric_filter=metric_filter) for group in groups
    ]
    sections.append(Prose([Paragraph(text=RESULTS_OUTRO)]))
    return BenchmarkReport(name="results", path=path, sections=sections, groups=groups)


def summary_report(
    comparison: ComparisonReport,
    groups: list[RunGroup],
    *,
    k: int = 10,
    stamp: str = "",
    path: Path = Path("README.md"),
    metric_filter: tuple[str, ...] = (),
) -> BenchmarkReport:
    """Build the README's results section: one headline per group at `k`.

    Spliced into the file's marked region: README.md is hand-written
    around it.
    """
    # imported here: sections use RunGroup/precision_pinned from this
    # module, so a top-level import would be circular
    from amb.reporting.sections import GroupSummary, Prose

    intro = f"Newest run per system at k={k} — full detail in [RESULTS.md](RESULTS.md)."
    lead: list[Block] = [Paragraph(text=stamp)] if stamp else []
    lead.append(Paragraph(text=intro))
    sections: list[Section] = [Prose(lead)]
    sections += [
        GroupSummary(group, comparison, k=k, level=3, metric_filter=metric_filter)
        for group in groups
    ]
    return BenchmarkReport(
        name="summary",
        path=path,
        sections=sections,
        marks=SUMMARY_MARKS,
        template="## Results\n\n{block}\n",
        groups=groups,
    )


# Every report this repo generates, by name. A new one — an ablation page,
# a per-provider page — is an entry here plus its builder; both CLI
# commands pick it up, `amb report` writing it and `amb plot all` drawing
# the charts it declares.
REPORTS: dict[str, Callable[..., BenchmarkReport]] = {
    "results": results_report,
    "summary": summary_report,
}


def build_reports(
    comparison: ComparisonReport,
    *,
    names: Sequence[str] = (),
    paths: dict[str, Path] | None = None,
    k: int = 10,
    stamp: str = "",
    dataset: str | None = None,
    variant: str | None = None,
    group_by: Sequence[str] = GROUP_BY,
    metric_filter: tuple[str, ...] = (),
) -> list[BenchmarkReport]:
    """Build the named reports (default: all) over the comparison's runs.

    The one entry point both `amb report` and `amb plot all` call, so the
    documents and the chart set are planned from the same objects and
    cannot disagree.

    Raises:
        ValueError: on an unknown report name, or when a group's runs
            disagree on a guarded identity field.
    """
    summaries = comparison.summaries
    if dataset is not None:
        summaries = [s for s in summaries if s.get("dataset") == dataset]
    if variant is not None:
        summaries = [s for s in summaries if (s.get("variant") or "") == variant]
    if not summaries:
        # naming a combination nothing ran is a mistake worth reporting, not
        # a document with an empty section in it
        scope = ", ".join(
            f"{name} {value!r}"
            for name, value in (("dataset", dataset), ("variant", variant))
            if value is not None
        )
        raise ValueError(f"no runs for {scope}" if scope else "no runs to report")
    scoped = ComparisonReport(summaries)
    groups = group_runs(summaries, group_by=group_by)

    selected = tuple(names) or tuple(REPORTS)
    unknown = [name for name in selected if name not in REPORTS]
    if unknown:
        raise ValueError(
            f"unknown report(s) {', '.join(unknown)}; known: {', '.join(REPORTS)}"
        )
    reports = []
    for name in selected:
        report = REPORTS[name](
            scoped, groups, k=k, stamp=stamp, metric_filter=metric_filter
        )
        if paths and name in paths:
            report.path = paths[name]
        reports.append(report)
    return reports
