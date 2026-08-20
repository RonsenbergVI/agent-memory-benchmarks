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

from datetime import UTC, datetime
from pathlib import Path

import click

from amb.constants import DEFAULT_RUNS_DIR
from amb.reporting import FORMATS, REPORTS, ComparisonReport, build_reports


@click.command()
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    help="directory of run data",
)
@click.option(
    "--latest", is_flag=True, help="only the newest run per dataset x system x models"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    metavar="PATH",
    help="write the `results` report to PATH, e.g. RESULTS.md. The file is "
    "generated whole — every word of it comes from the runs, so anything "
    "hand-written there is overwritten",
)
@click.option(
    "--summary",
    type=click.Path(path_type=Path),
    metavar="PATH",
    help="write the `summary` report into PATH's marked section, e.g. "
    "README.md, whose surrounding prose is hand-written and preserved",
)
@click.option(
    "--report",
    "extra",
    multiple=True,
    metavar="NAME[=PATH]",
    help=f"write the named report ({', '.join(REPORTS)}), "
    "optionally to PATH instead of its default (repeatable). --output and "
    "--summary are the shorthands for the two built-in ones",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(FORMATS),
    default="markdown",
    show_default=True,
    help="output format for every generated report",
)
@click.option(
    "--k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="the k the --summary table compares at",
)
@click.option(
    "--dataset",
    help="which dataset's runs to report (default: every dataset present, "
    "one section each)",
)
@click.option(
    "--variant",
    help="which dataset variant to report, when a dataset has more than one "
    "(default: every variant present, one section each)",
)
@click.option(
    "--tag",
    metavar="TAG",
    help="stamp the generated reports with this release tag",
)
@click.option(
    "--commit",
    metavar="SHA",
    help="stamp the generated reports with the commit that produced them "
    "(shown shortened to 7 characters)",
)
def report(  # noqa: PLR0913 - one option per output/filter, all independent
    directory: Path,
    latest: bool,
    output: Path | None,
    summary: Path | None,
    extra: tuple[str, ...],
    fmt: str,
    k: int,
    dataset: str | None,
    variant: str | None,
    tag: str | None,
    commit: str | None,
) -> None:
    """Generate the comparison reports from the saved runs.

    Named nothing to write, it prints the full table to stdout. Otherwise
    each requested report is generated from `amb.reporting`: the `results`
    report (RESULTS.md) is written whole, since every part of it is derived
    from the runs; the `summary` report (README.md) is spliced into its
    marked section, since hand-written prose surrounds it. Both cover every
    (dataset, variant) present — one section each — unless
    --dataset/--variant narrow them, and both declare the charts `amb plot all`
    draws, so documents and images cannot drift apart.

    Raises:
        click.ClickException: if no summaries were found under the
            directory, an unknown report was named, --dataset/--variant name
            a combination with no runs, or a section's runs span several
            modes.
    """
    requested: dict[str, Path | None] = {}
    if output:
        requested["results"] = output
    if summary:
        requested["summary"] = summary
    for item in extra:
        name, _, override = item.partition("=")
        requested[name] = Path(override) if override else None

    comparison = ComparisonReport.collect(directory)
    if latest:
        comparison = comparison.latest()
    if not comparison.summaries:
        raise click.ClickException(f"no summaries under {directory}/")

    stamp = ""
    if tag or commit:
        date = datetime.now(UTC).date().isoformat()
        ref = f"`{tag}`" if tag else ""
        if commit:
            sha = f"commit `{commit[:7]}`"
            ref = f"{ref} ({sha})" if ref else sha
        stamp = f"Results from {ref}, run {date}."

    if not requested:
        click.echo(comparison.to_markdown())
        return
    try:
        reports = build_reports(
            comparison,
            names=tuple(requested),
            paths={n: p for n, p in requested.items() if p is not None},
            k=k,
            stamp=stamp,
            dataset=dataset,
            variant=variant,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    for built in reports:
        built.write(fmt)
        click.echo(
            f"wrote {built.path} ({len(built.groups)} section(s), "
            f"{len(comparison.summaries)} runs, k={k})"
        )
