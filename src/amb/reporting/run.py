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
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from amb.base import Metric, Report
from amb.constants import DEFAULT_RUNS_DIR
from amb.contracts import Run
from amb.metrics import default_category_metrics, default_metrics

if TYPE_CHECKING:
    from pydantic_ai.models import Model

# the summary table's columns: (summary key, column label, format)
SUMMARY_COLUMNS = (
    ("retrieval_precision", "precision", "{:.3f}"),
    ("retrieval_recall", "recall", "{:.3f}"),
    ("retrieval_f1", "F1", "{:.3f}"),
    ("memory_tokens_total", "memory tokens", "{:,.0f}"),
)


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a header and formatted rows as a GitHub markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


class RunReport(Report):
    """The metrics report over one run's raw data.

    Consumes the Run a benchmark returned; the metrics are an input,
    defaulting to the standard set. `metric_set` holds instances (reset
    before each pass); `category_metric_set` is a factory because every
    question category needs its own instances.
    """

    def __init__(
        self,
        run: Run,
        metric_set: list[Metric] | None = None,
        category_metric_set: Callable[[], list[Metric]] | None = None,
    ) -> None:
        """Bind the run's raw data to the metrics that will score it."""
        self.run = run
        self.metric_set = metric_set
        self.category_metric_set = category_metric_set

    def summary(self) -> dict:
        """Score the run: one pass of every record through the metric set.

        Ingest stats then rows; each metric grabs the fields that apply to it
        and its result lands under its (possibly dotted) name.
        """
        metric_set = (
            self.metric_set if self.metric_set is not None else default_metrics()
        )
        for metric in metric_set:
            metric.reset_state()
        for observation in (*self.run.ingestion_records, *self.run.question_records):
            for metric in metric_set:
                metric.update_state(observation)

        category_factory = self.category_metric_set or default_category_metrics
        by_category: dict[str, list[Metric]] = {}
        for row in self.run.question_records:
            bucket = by_category.setdefault(
                row.category or "uncategorized", category_factory()
            )
            for metric in bucket:
                metric.update_state(row)

        summary: dict = {
            "run_id": self.run.run_id,
            "system": self.run.system,
            "system_version": self.run.system_version,
            "dataset": self.run.dataset,
            "variant": self.run.variant,
            "mode": self.run.mode,
            "k": self.run.k,
            "model": self.run.model,
            "judge_model": self.run.judge_model,
            "num_samples": len(self.run.ingestion_records),
            "num_questions": len(self.run.question_records),
        }
        # the memory system's own models and any remaining explicit --param
        # overrides: part of identity, so variants coexist as rows
        if self.run.ingestion_model:
            summary["ingestion_model"] = self.run.ingestion_model
        if self.run.embedding_model:
            summary["embedding_model"] = self.run.embedding_model
        if self.run.system_params:
            summary["system_params"] = self.run.system_params
        if self.run.max_turns is not None:
            # marks a truncated run: only part of the corpus was ingested, so
            # these scores are not full-run numbers
            summary["max_turns"] = self.run.max_turns
        if self.run.sample_seed is not None:
            # which conversations were drawn is part of what was measured
            summary["sample_seed"] = self.run.sample_seed
        for metric in metric_set:
            if metric.count or metric.report_when_empty:
                self._place(summary, metric.name, metric.result())
        # headline cost: one top-level float so it shows up as a comparison
        # column whenever a usage-tracking callback reported spend
        section = summary.get("memory_tokens")
        if isinstance(section, dict):
            ingested: dict = section.get("ingest", {})
            summary["memory_tokens_total"] = float(
                sum(v for k, v in ingested.items() if k.endswith("_tokens"))
                + section.get("search_total", 0)
            )
        summary["by_category"] = {
            category: {
                m.name: m.result() for m in bucket if m.count or m.report_when_empty
            }
            for category, bucket in sorted(by_category.items())
        }
        return summary

    @staticmethod
    def _place(target: dict, name: str, value: object) -> None:
        """Set a (possibly dotted) metric name into nested summary dicts."""
        *sections, leaf = name.split(".")
        for section in sections:
            target = target.setdefault(section, {})
        target[leaf] = value

    # -- evaluation ------------------------------------------------------

    def judge(self, model: "str | Model", force: bool = False) -> int:
        """Grade predicted answers against gold, writing onto the rows.

        An evaluation step, deliberately outside the run loop: a saved run
        can be (re-)judged — different judge model, tightened rubric —
        without re-running the memory system. Returns how many rows were
        judged; rows without a prediction or a gold answer are skipped, and
        already-judged rows are kept unless `force`.
        """
        # imported lazily so judging is the only path that needs LLM deps
        from amb.agent import judge_answer

        judged = 0
        for row in self.run.question_records:
            if not row.predicted_answer or not row.gold_answer:
                continue
            if row.judge_correct is not None and not force:
                continue
            judgment = judge_answer(
                row.question or "", row.predicted_answer, row.gold_answer, model
            )
            row.judge_correct = judgment.correct
            row.judge_reasoning = judgment.reasoning
            judged += 1
        if judged:
            self.run.judge_model = str(model)
        return judged

    # -- persistence -----------------------------------------------------

    def run_dir(self, root: Path | None = None) -> Path:
        """This run's own directory under the report root.

        Laid out as `<root>/<dataset>[/<variant>]/<system>/<mode>/<run_id>`.
        The mode level keeps the two experiment kinds apart on disk the same
        way they are kept apart in every comparison. The variant level only
        appears when the run named one (`--variant`) — most datasets have
        none, but a dataset with several (e.g. LongMemEval's oracle/s/m) is
        a different experiment per variant, and without this level they'd
        share one directory and silently collide in `--latest`.
        """
        parts = [self.run.dataset]
        if self.run.variant:
            parts.append(self.run.variant)
        parts += [self.run.system, self.run.mode, self.run.run_id]
        return (root or DEFAULT_RUNS_DIR).joinpath(*parts)

    def save(self, root: Path | None = None) -> Path:
        """Write rows, ingest stats, and the summary; return the directory."""
        out = self.run_dir(root)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "results.jsonl").open("w") as sink:
            for row in self.run.question_records:
                sink.write(json.dumps(row.model_dump(exclude_none=True)) + "\n")
        (out / "ingest.json").write_text(
            json.dumps(
                [s.model_dump(exclude_none=True) for s in self.run.ingestion_records],
                indent=2,
            )
        )
        (out / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        return out

    @classmethod
    def load(cls, run_dir: Path) -> "RunReport":
        """Rebuild a report from a saved run directory, ready to re-score."""
        summary = json.loads((run_dir / "summary.json").read_text())
        rows = [
            json.loads(line)
            for line in (run_dir / "results.jsonl").read_text().splitlines()
        ]
        ingest_path = run_dir / "ingest.json"
        ingest = json.loads(ingest_path.read_text()) if ingest_path.exists() else []
        return cls(
            Run(
                run_id=summary["run_id"],
                system=summary["system"],
                dataset=summary["dataset"],
                variant=summary.get("variant"),
                mode=summary.get("mode", "direct"),
                k=summary.get("k", 10),
                model=summary.get("model"),
                judge_model=summary.get("judge_model"),
                ingestion_model=summary.get("ingestion_model"),
                embedding_model=summary.get("embedding_model"),
                system_params=summary.get("system_params") or {},
                max_turns=summary.get("max_turns"),
                sample_seed=summary.get("sample_seed"),
                system_version=summary.get("system_version"),
                question_records=rows,
                ingestion_records=ingest,
            )
        )

    # -- rendering ---------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the run's summary as a two-column markdown table.

        Nested sections (dotted metric names, by_category) are shown as
        their dict text — the cross-run tables live on ComparisonReport.
        """
        return markdown_table(
            ["field", "value"],
            [[key, str(value)] for key, value in self.summary().items()],
        )


class ComparisonReport(Report):
    """The final cross-system comparison, aggregated from run summaries.

    Deliberately unaware of providers, datasets, and metrics: it loads
    whatever summaries exist and the data itself says who is what — identity
    comes from each summary's ``system``/``dataset`` fields, and every
    top-level float in a summary is treated as a reported metric, so new
    metrics and new systems appear in the comparison without code changes.
    """

    # max_turns is part of identity: a truncated smoke run is a different
    # thing from a full run and must never displace one in latest(). The
    # system's own models and --param overrides are identity too: a fraise
    # run with an embedder is a different subject than one without. So is
    # k: recall@1 and recall@10 measure the same system at different
    # retrieval budgets, and a --k sweep must land as separate rows.
    IDENTITY = (
        "dataset",
        "variant",
        "system",
        "system_version",
        "mode",
        "k",
        "ingestion_model",
        "embedding_model",
        "system_params",
        "model",
        "judge_model",
        "max_turns",
        "sample_seed",
        "num_questions",
    )

    @staticmethod
    def identity_label(summary: dict, column: str) -> str:
        """One identity cell as text; params render as `k=v` pairs."""
        value = summary.get(column)
        if value is None:
            return ""
        if isinstance(value, dict):
            return " ".join(f"{k}={v}" for k, v in sorted(value.items()))
        return str(value)

    def __init__(self, summaries: list[dict]) -> None:
        """Hold the summaries in a stable dataset/system/run order."""
        self.summaries = sorted(
            summaries,
            key=lambda s: (
                s.get("dataset", ""),
                s.get("system", ""),
                s.get("run_id", ""),
            ),
        )

    @classmethod
    def collect(cls, root: Path | None = None) -> "ComparisonReport":
        """Load every summary under root, wherever it sits.

        Identity is read from each file's content, not from the directory
        layout.
        """
        root = root or DEFAULT_RUNS_DIR
        summaries = []
        for path in sorted(root.rglob("summary.json")):
            data = json.loads(path.read_text())
            if {"system", "dataset"} <= data.keys():
                summaries.append(data)
        return cls(summaries)

    def latest(self) -> "ComparisonReport":
        """Only the newest run per identity (dataset, system, mode, models)."""
        newest: dict[tuple, dict] = {}
        for s in self.summaries:
            key = tuple(
                self.identity_label(s, col)
                for col in self.IDENTITY
                if col != "num_questions"
            )
            if key not in newest or (s.get("run_id") or "") > (
                newest[key].get("run_id") or ""
            ):
                newest[key] = s
        return ComparisonReport(list(newest.values()))

    def metric_columns(self) -> list[str]:
        """Discovered from the data: any top-level float is a metric."""
        keys: set[str] = set()
        for s in self.summaries:
            keys.update(k for k, v in s.items() if isinstance(v, float))
        return sorted(keys)

    def to_dict(self) -> dict:
        """Comparison keyed by dataset -> system -> latest summary."""
        table: dict[str, dict[str, dict]] = defaultdict(dict)
        for s in self.summaries:  # sorted, so the latest run_id wins
            table[s["dataset"]][s["system"]] = s
        return dict(table)

    def summary_table(
        self, k: int = 10, dataset: str | None = None, variant: str | None = None
    ) -> tuple[list[str], list[list[str]]]:
        """Build the compact table: newest run per system at one k.

        The full-detail table (`to_markdown`) keeps every identity row
        distinct; this one answers "how do the systems compare" in one row
        per system, best retrieval F1 first, for the README. `dataset`
        scopes it to one dataset — comparing "newest per system" across
        several datasets at once would silently mix unrelated exams into
        one ranking, so the caller must resolve one first (see
        `to_markdown`'s mixed-dataset guard for the same concern).
        `variant` scopes it further, to one dataset variant (e.g.
        LongMemEval's oracle/s/m are different experiments, not different
        runs of the same one) — required whenever the dataset has more
        than one variant present.

        Returns:
            The header labels and the formatted rows, ranked best F1 first.
            Formatting into a document is the renderer's job.
        """
        newest: dict[str, dict] = {}
        for s in self.summaries:
            if s.get("k") != k:
                continue
            if dataset is not None and s.get("dataset") != dataset:
                continue
            if variant is not None and s.get("variant") != variant:
                continue
            system = str(s.get("system", "?"))
            if system not in newest or (s.get("run_id") or "") > (
                newest[system].get("run_id") or ""
            ):
                newest[system] = s
        header = ["system", "version", *(label for _, label, _ in SUMMARY_COLUMNS)]
        header += ["p50 search (s)"]
        ranked = sorted(
            newest.values(),
            key=lambda s: (-s.get("retrieval_f1", float("-inf")), s.get("system", "")),
        )
        rows = []
        for s in ranked:
            cells = [str(s.get("system", "?")), str(s.get("system_version") or "")]
            cells += [
                fmt.format(s[key]) if key in s else ""
                for key, _, fmt in SUMMARY_COLUMNS
            ]
            latency = s.get("search_latency", {})
            cells += [f"{latency.get('p50_s', 0):.4f}" if latency else ""]
            rows.append(cells)
        return header, rows

    def to_summary_markdown(
        self, k: int = 10, dataset: str | None = None, variant: str | None = None
    ) -> str:
        """Render `summary_table` as a markdown table."""
        return markdown_table(*self.summary_table(k, dataset, variant))

    def table(self) -> tuple[list[str], list[list[str]]]:
        """Build the full comparison: one row per run, every identity column.

        Every metric column is session-level and populated for every
        system — one exam, fully comparable.

        Returns:
            The header labels and the formatted rows, in the report's
            stable dataset/system/run order.
        """
        metric_cols = self.metric_columns()
        has_latency = any("search_latency" in s for s in self.summaries)
        header = [*self.IDENTITY, *metric_cols]
        if has_latency:
            header += ["search_p50_s", "search_p95_s"]
        rows = []
        for s in self.summaries:
            cells = [self.identity_label(s, col) for col in self.IDENTITY]
            cells += [f"{s[k]:.3f}" if k in s else "" for k in metric_cols]
            if has_latency:
                latency = s.get("search_latency", {})
                cells += [
                    f"{latency.get('p50_s', 0):.4f}",
                    f"{latency.get('p95_s', 0):.4f}",
                ]
            rows.append(cells)
        return header, rows

    def to_markdown(self) -> str:
        """Render `table` as a markdown table."""
        return markdown_table(*self.table())
