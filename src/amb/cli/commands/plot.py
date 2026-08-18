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

import click

from amb import documents, plotting
from amb.constants import DEFAULT_REPORT_DIR
from amb.documents import headline
from amb.reporting import ComparisonReport


@click.group()
def plot() -> None:
    """Draw charts from the saved runs."""


@plot.command()
@click.option(
    "--x",
    "x_metric",
    required=True,
    metavar="METRIC",
    help="metric on the x axis, e.g. search_latency.p50_s",
)
@click.option(
    "--y",
    "y_metric",
    required=True,
    metavar="METRIC",
    help="metric on the y axis, e.g. retrieval_recall",
)
@click.option("--dataset", help="which dataset's runs to plot")
@click.option(
    "--k",
    type=click.IntRange(min=1),
    metavar="N",
    help="which k's runs to plot, when the reports span a --k sweep",
)
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=DEFAULT_REPORT_DIR,
    show_default=True,
    help="directory of run reports",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("plot.png"),
    show_default=True,
    help="image to write (.png or .svg)",
)
@click.option("--title", help="chart title; defaults to naming the dataset")
@click.option("--dark", is_flag=True, help="render on the dark chart surface")
@click.option(
    "--list-metrics",
    is_flag=True,
    help="list the metric names available to plot, then exit",
)
def scatter(  # noqa: PLR0913 - one option per axis/filter, all independent
    x_metric: str,
    y_metric: str,
    dataset: str | None,
    k: int | None,
    directory: Path,
    output: Path,
    title: str | None,
    dark: bool,
    list_metrics: bool,
) -> None:
    """Plot one dot per memory system, with a metric on each axis.

    Raises:
        click.ClickException: if no runs, no metrics, mixed datasets, or
            mixed k values.
    """
    comparison = ComparisonReport.collect(directory).latest()
    summaries = comparison.summaries
    if dataset:
        summaries = [s for s in summaries if s.get("dataset") == dataset]
    if k is not None:
        summaries = [s for s in summaries if s.get("k") == k]
    if not summaries:
        raise click.ClickException(f"no runs to plot under {directory}/")

    if list_metrics:
        click.echo("\n".join(plotting.available_metrics(summaries)))
        return

    # points from different datasets are not comparable on one pair of axes
    datasets = {s.get("dataset") for s in summaries}
    if len(datasets) > 1:
        raise click.ClickException(
            f"runs span several datasets ({', '.join(sorted(map(str, datasets)))}); "
            "pass --dataset to pick one"
        )
    # neither are runs at different k: recall@1 beside recall@10 on one
    # axis compares budgets, not systems
    ks = {s.get("k") for s in summaries}
    if len(ks) > 1:
        raise click.ClickException(
            f"runs span several k values ({', '.join(map(str, sorted(ks, key=str)))});"
            " pass --k to pick one"
        )

    points = plotting.collect_points(summaries, x_metric, y_metric)
    if not points:
        available = plotting.available_metrics(summaries)
        raise click.ClickException(
            f"no run has both {x_metric!r} and {y_metric!r}. Available:\n  "
            + "\n  ".join(available)
        )

    k_value = ks.pop()
    subtitle = str(datasets.pop())
    if k_value is not None:
        subtitle += f" · k={k_value}"
    written = plotting.scatter(
        points,
        x_label=x_metric,
        y_label=y_metric,
        output=output,
        title=title
        or headline(f"{plotting.pretty(y_metric)} vs {plotting.pretty(x_metric)}"),
        subtitle=subtitle,
        dark=dark,
    )
    click.echo(f"wrote {written} ({len(points)} systems)")


@plot.command(name="k")
@click.option(
    "--metric",
    "-m",
    required=True,
    metavar="METRIC",
    help="metric on the y axis, e.g. retrieval_recall; x is always k",
)
@click.option("--dataset", help="which dataset's runs to plot")
@click.option(
    "--mode",
    type=click.Choice(["direct", "agentic"]),
    help="which mode's runs to plot, when both exist",
)
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=DEFAULT_REPORT_DIR,
    show_default=True,
    help="directory of run reports",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("plot.png"),
    show_default=True,
    help="image to write (.png or .svg)",
)
@click.option("--title", help="chart title; defaults to naming the dataset")
@click.option("--dark", is_flag=True, help="render on the dark chart surface")
def plot_k(  # noqa: PLR0913 - one option per axis/filter, all independent
    metric: str,
    dataset: str | None,
    mode: str | None,
    directory: Path,
    output: Path,
    title: str | None,
    dark: bool,
) -> None:
    """Plot one metric across a --k sweep, one line per memory system.

    Systems run at a single k appear as lone dots; their lines grow as
    sweeps accumulate.

    Raises:
        click.ClickException: if no runs, an unknown metric, mixed
            datasets, or mixed modes.
    """
    comparison = ComparisonReport.collect(directory).latest()
    summaries = comparison.summaries
    if dataset:
        summaries = [s for s in summaries if s.get("dataset") == dataset]
    if mode:
        summaries = [s for s in summaries if s.get("mode") == mode]
    if not summaries:
        raise click.ClickException(f"no runs to plot under {directory}/")

    # points from different datasets are not comparable on one pair of axes
    datasets = {s.get("dataset") for s in summaries}
    if len(datasets) > 1:
        raise click.ClickException(
            f"runs span several datasets ({', '.join(sorted(map(str, datasets)))}); "
            "pass --dataset to pick one"
        )
    # neither are the two experiment kinds: a direct and an agentic run of
    # one system are different subjects, not two points on one line
    modes = {s.get("mode") for s in summaries}
    if len(modes) > 1:
        raise click.ClickException(
            f"runs span several modes ({', '.join(sorted(map(str, modes)))}); "
            "pass --mode to pick one"
        )

    series = plotting.collect_series(summaries, metric)
    if not series:
        available = plotting.available_metrics(summaries)
        raise click.ClickException(
            f"no run reports {metric!r}. Available:\n  " + "\n  ".join(available)
        )

    written = plotting.lines(
        series,
        x_label="k (hits requested per query)",
        y_label=metric,
        output=output,
        title=title or headline(f"{plotting.pretty(metric)} vs k"),
        subtitle=f"{datasets.pop()} · {modes.pop()}",
        dark=dark,
    )
    click.echo(f"wrote {written} ({len(series)} systems)")


@plot.command(name="all")
@click.option(
    "--dataset", help="which dataset's runs to plot (default: every one present)"
)
@click.option(
    "--variant",
    help="which dataset variant to plot (default: every one present, each "
    "into its own directory)",
)
@click.option(
    "--mode",
    type=click.Choice(["direct", "agentic"]),
    help="which mode's runs to plot, when both exist",
)
@click.option(
    "--k",
    type=click.IntRange(min=1),
    metavar="N",
    help="restrict the single-k charts to this k [default: every k the runs "
    "used]; the k-sweep charts always span the whole sweep",
)
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=DEFAULT_REPORT_DIR,
    show_default=True,
    help="directory of run reports",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="write every image into this one directory instead of the "
    "plots/<dataset>[/<variant>] layout the reports link; only valid when the "
    "selection resolves to a single chart set",
)
@click.option(
    "--metric",
    "metric_filter",
    type=click.Choice(["precision", "recall", "f1"]),
    multiple=True,
    help="restrict to these retrieval metrics (default: all three, or just "
    "recall when every run in scope reports precision exactly 1.0 — e.g. "
    "LongMemEval's oracle variant, whose haystack ships only evidence). "
    "Repeatable; overrides the auto-detection either way",
)
@click.option("--dark", is_flag=True, help="render on the dark chart surface")
def draw_all(  # noqa: PLR0913 - one option per filter, all independent
    dataset: str | None,
    variant: str | None,
    mode: str | None,
    k: int | None,
    directory: Path,
    out: Path | None,
    metric_filter: tuple[str, ...],
    dark: bool,
) -> None:
    """Draw every chart the generated reports declare.

    The chart set is not defined here: `amb report`'s documents declare the
    figures they show, and this command draws exactly those, into
    `plots/<dataset>/` (or `plots/<dataset>/<variant>/` for a dataset whose
    variants are separate experiments). So a chart can never be missing
    from a document, nor written for one nobody shows. Every dataset and
    variant present is covered in one call — narrow it with the filters.

    Per metric (precision/recall/f1 by default — see --metric and its
    auto-detection): `session_<metric>_k<k>` bars comparing systems, and
    `k_<metric>` lines across the whole sweep. Cross metric:
    `tokens_<metric>_k<k>` and `latency_<metric>_k<k>` trade-off scatters,
    plus `tokens_latency_k<k>`. The single-k sets are written for every k
    the runs used, each carrying its k in the filename.

    Raises:
        click.ClickException: if no runs, a group spans several modes, or
            --out names one directory for several chart sets.
    """
    comparison = ComparisonReport.collect(directory).latest()
    if mode:
        comparison = ComparisonReport(
            [s for s in comparison.summaries if s.get("mode") == mode]
        )
    if not comparison.summaries:
        raise click.ClickException(f"no runs to plot under {directory}/")
    try:
        reports = documents.build_reports(
            comparison,
            dataset=dataset,
            variant=variant,
            metric_filter=metric_filter,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    charts = documents.plan_charts(reports)
    if out is not None:
        # chart filenames are not namespaced by dataset, so collapsing two
        # sets into one directory would have them overwrite each other
        sets = sorted({str(chart.out_dir) for chart in charts})
        if len(sets) > 1:
            raise click.ClickException(
                f"--out is one directory but the selection spans {len(sets)} "
                f"chart sets ({', '.join(sets)}); pass --dataset/--variant to "
                "pick one, or drop --out to write each into its own directory"
            )
        charts = documents.plan_charts(reports, out=out)
    if k is not None:
        # the sweep charts (k is None) span every k, so they always apply
        charts = [chart for chart in charts if chart.k in (None, k)]

    written = [path for chart in charts if (path := chart.draw(dark)) is not None]
    for path in written:
        click.echo(f"wrote {path}")
    skipped = sum(report.skipped for report in reports)
    tail = f", {skipped} skipped with no data" if skipped else ""
    click.echo(f"{len(written)} charts written{tail}")
