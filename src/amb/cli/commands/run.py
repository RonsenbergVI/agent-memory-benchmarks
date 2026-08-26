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

import json
from pathlib import Path

import click

from amb.constants import DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR
from amb.registry import get_benchmark
from amb.reporting import RunReport
from amb.runner import RunConfig, Runner


class ParamType(click.ParamType):
    """A `KEY=VALUE` memory-system parameter.

    Values are coerced (`reasoning=false` arrives as False, not "false");
    only the first `=` splits, so URLs survive intact.
    """

    name = "key=value"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> tuple[str, object]:
        """Split KEY=VALUE on the first `=` and coerce the value."""
        key, sep, raw = value.partition("=")
        if not sep:
            self.fail(f"{value!r} is not KEY=VALUE", param, ctx)
        return key, self.coerce(raw)

    @staticmethod
    def coerce(raw: str) -> object:
        """Turn a CLI string into a bool, None, int, float, or string."""
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        if raw.lower() in ("none", "null"):
            return None
        for cast in (int, float):
            try:
                return cast(raw)
            except ValueError:
                pass
        return raw


PARAM = ParamType()


@click.command()
@click.option("--system", required=True, help="memory system to benchmark")
@click.option("--dataset", required=True, help="dataset to run against")
@click.option("--variant", help="dataset variant; defaults to the loader's own")
@click.option(
    "--mode",
    type=click.Choice(["direct", "agentic"]),
    default="direct",
    show_default=True,
    help="direct: the harness drives ingestion and search; "
    "agentic: a model drives both through the system's own tools. "
    "Answers are generated iff --model is set (mandatory for agentic)",
)
@click.option(
    "--k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="hits requested per query",
)
@click.option("--limit", type=click.IntRange(min=1), help="max conversations")
@click.option(
    "--seed",
    "sample_seed",
    type=int,
    metavar="N",
    help="draw --limit conversations at random, reproducibly, "
    "instead of taking the first ones",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="conversations ingested/queried in parallel "
    "(sessions within each stay ordered)",
)
@click.option(
    "--keep", is_flag=True, help="skip teardown so the store keeps this run's memories"
)
@click.option(
    "--reuse",
    is_flag=True,
    help="skip ingestion and query the store as-is (after a --keep "
    "run of the same scope)",
)
@click.option(
    "--turns",
    "max_turns",
    type=click.IntRange(min=1),
    metavar="N",
    help="ingest at most N turns per conversation (smoke runs). Sessions are "
    "taken in order until the budget runs out; questions whose evidence falls "
    "outside it are skipped. Truncated runs are marked and never compared "
    "against full ones",
)
@click.option(
    "--questions",
    "max_questions",
    type=click.IntRange(min=1),
    help="max questions asked per conversation (workload, not corpus size — "
    "asked after ingestion, so it does not make ingestion cheaper)",
)
@click.option(
    "--model",
    help="answer model (pydantic-ai id, e.g. openai:gpt-5-mini). No default: "
    "direct runs skip answer generation without it; agentic runs require it",
)
@click.option(
    "--judge",
    is_flag=True,
    help="judge answers right after the run (same step as `amb judge`)",
)
@click.option(
    "--judge-model", help="judge model (pydantic-ai id; defaults to the answer model)"
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="dataset cache directory",
)
@click.option(
    "--runs-dir",
    "out",
    type=click.Path(path_type=Path),
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    help="where this run's data is written",
)
@click.option(
    "--param",
    "params",
    type=PARAM,
    multiple=True,
    metavar="KEY=VALUE",
    help="set a memory-system parameter, e.g. model=gpt-5-mini or "
    "embedding_model=text-embedding-3-small (repeatable; true/false and "
    "numbers are coerced)",
)
def run(system: str, params: tuple[tuple[str, object], ...], **options) -> None:
    """Run one system against one dataset.

    Raises:
        click.ClickException: with --judge but no judge model to use.
    """
    # --param belongs to the system under test; the rest is run execution
    benchmark = get_benchmark(system).benchmark_class()(params=dict(params))
    config = RunConfig(**options)
    data = Runner(benchmark, config).run()
    report = RunReport(data)
    if config.judge:
        # the same evaluation step as `amb judge`, applied before saving
        judge_model = config.judge_model or config.model
        if not judge_model:
            raise click.ClickException("--judge needs --judge-model or --model")
        report.judge(judge_model)
    report.save(config.out)
    click.echo(json.dumps(report.summary(), indent=2))
