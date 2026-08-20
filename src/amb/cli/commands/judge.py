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

from amb.reporting import RunReport


@click.command()
@click.argument(
    "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--model", help="judge model (pydantic-ai id); defaults to the run's answer model"
)
@click.option(
    "--force", is_flag=True, help="re-judge rows that already carry a judgment"
)
def judge(run_dir: Path, model: str | None, force: bool) -> None:
    """Grade a saved run's answers with an LLM judge, updating it in place.

    RUN_DIR is a run's data directory
    (runs/<dataset>/<system>/<mode>/<run-id>, or
    runs/<dataset>/<variant>/<system>/<mode>/<run-id> for a run with
    --variant).
    Judging is an evaluation step: it needs no database and no re-run, so a
    run can be re-judged with a different model at any time.

    Raises:
        click.ClickException: when no judge model is known, or the run has
            no predicted answers to grade.
    """
    report = RunReport.load(run_dir)
    judge_model = model or report.run.model
    if not judge_model:
        raise click.ClickException(
            "this run recorded no answer model; pass --model to pick the judge"
        )
    judged = report.judge(judge_model, force=force)
    if not judged and not any(r.predicted_answer for r in report.run.question_records):
        raise click.ClickException(
            "no predicted answers to grade: this run never generated answers "
            "(direct mode without --model)"
        )
    # variant runs live one level deeper (<root>/<dataset>/<variant>/...),
    # so the runs root sits one more parent up
    report.save(run_dir.parents[4 if report.run.variant else 3])
    click.echo(f"judged {judged} answers with {judge_model} -> {run_dir}")
