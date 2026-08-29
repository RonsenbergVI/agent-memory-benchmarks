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


def run_date(run_id: str | None) -> str:
    """A run's date, read off its timestamp id (YYYY-MM-DD, or '')."""
    if run_id and len(run_id) >= 8 and run_id[:8].isdigit():
        return f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
    return ""


def category_counts(summary: dict, root: Path | None = None) -> dict[str, int]:
    """Question counts per category, read from the run's saved rows.

    From results.jsonl rather than the summary, so runs recorded before the
    category table existed still fill their n column; {} when the rows are
    not on disk (the summary travelled without its run directory).
    """
    parts = [summary.get("dataset", "")]
    if summary.get("variant"):
        parts.append(str(summary["variant"]))
    parts += [
        str(summary.get("system", "")),
        str(summary.get("mode", "direct")),
        str(summary.get("run_id", "")),
    ]
    path = (root or DEFAULT_RUNS_DIR).joinpath(*parts, "results.jsonl")
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text().splitlines():
        category = json.loads(line).get("category") or "uncategorized"
        counts[category] = counts.get(category, 0) + 1
    return counts


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

    `metric_set` holds instances (reset before each pass); `category_metric_set`
    is a factory because every question category needs its own instances.
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

        Each metric's result lands under its (possibly dotted) name.
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
        # the system's own models and --param overrides are identity: variants
        # must coexist as rows
        if self.run.ingestion_model:
            summary["ingestion_model"] = self.run.ingestion_model
        if self.run.embedding_model:
            summary["embedding_model"] = self.run.embedding_model
        if self.run.usage_coverage != "full":
            summary["usage_coverage"] = self.run.usage_coverage
        if self.run.system_params:
            summary["system_params"] = self.run.system_params
        if self.run.max_turns is not None:
            # marks a truncated run: scores are not full-run numbers
            summary["max_turns"] = self.run.max_turns
        if self.run.sample_seed is not None:
            # which conversations were drawn is part of what was measured
            summary["sample_seed"] = self.run.sample_seed
        if self.run.workers > 1:
            # latencies measured under N-way contention: a different experiment
            # from single-worker rows, never blended
            summary["workers"] = self.run.workers
        for metric in metric_set:
            if metric.count or metric.report_when_empty:
                self._place(summary, metric.name, metric.result())
        # headline cost as one top-level float; omitted entirely when coverage
        # is "none" — an unobservable spend reported as 0 would read as "free"
        section = summary.get("memory_tokens")
        if isinstance(section, dict) and self.run.usage_coverage != "none":
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

        Outside the run loop so a saved run can be re-judged without re-running
        the memory system. Returns rows judged; rows missing a prediction or
        gold answer are skipped, already-judged rows kept unless `force`.
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
        """`<root>/<dataset>[/<variant>]/<system>/<mode>/<run_id>`.

        The variant level appears only when the run named one; without it,
        variants (e.g. LongMemEval oracle/s/m — a different experiment each)
        would share one directory and silently collide in `--latest`.
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
                usage_coverage=summary.get("usage_coverage", "full"),
                system_params=summary.get("system_params") or {},
                max_turns=summary.get("max_turns"),
                sample_seed=summary.get("sample_seed"),
                workers=summary.get("workers", 1),
                system_version=summary.get("system_version"),
                question_records=rows,
                ingestion_records=ingest,
            )
        )

    # -- rendering ---------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the run's summary as a two-column markdown table.

        Nested sections show as dict text; cross-run tables live on
        ComparisonReport.
        """
        return markdown_table(
            ["field", "value"],
            [[key, str(value)] for key, value in self.summary().items()],
        )


class ComparisonReport(Report):
    """The final cross-system comparison, aggregated from run summaries.

    Identity comes from each summary's fields and every top-level float is a
    reported metric, so new metrics and systems appear without code changes.
    """

    # max_turns is identity: a truncated smoke run must never displace a full
    # run in latest(). So are the system's own models/--param overrides, and k:
    # a --k sweep must land as separate rows.
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
        "workers",
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
        """Load every summary under root.

        Identity is read from file content, not directory layout.
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

    def latest_at_k(
        self, k: int = 10, dataset: str | None = None, variant: str | None = None
    ) -> list[dict]:
        """Newest run per system at one k, ranked best F1 first.

        `dataset` must scope to one dataset — ranking "newest per system"
        across several would mix unrelated exams. `variant` scopes further and
        is required whenever the dataset has more than one variant present.
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
        return sorted(
            newest.values(),
            key=lambda s: (-s.get("retrieval_f1", float("-inf")), s.get("system", "")),
        )

    def category_table(
        self,
        k: int = 10,
        dataset: str | None = None,
        variant: str | None = None,
        metric: str = "retrieval_recall",
    ) -> tuple[list[str], list[list[str]]]:
        """One retrieval metric by question category: newest run per system at k.

        Columns rank like the summary table (best overall F1 first); `n` is
        the category's question count, read from the runs' saved rows and
        taken as the largest across systems so a run that dropped questions
        cannot shrink the battery it is compared on.

        Returns:
            The header labels and rows; no rows when no run reports categories.
        """
        ranked = [
            s for s in self.latest_at_k(k, dataset, variant) if s.get("by_category")
        ]
        if not ranked:
            return ["category", "n"], []
        categories = sorted({c for s in ranked for c in s["by_category"]})
        counts: dict[str, int] = {}
        for s in ranked:
            for category, n in category_counts(s).items():
                counts[category] = max(counts.get(category, 0), n)
        header = ["category", "n", *(str(s.get("system", "?")) for s in ranked)]
        rows = [
            [
                category,
                str(counts.get(category, "")),
                *(
                    f"{cell[metric]:.3f}"
                    if metric in (cell := s["by_category"].get(category, {}))
                    else ""
                    for s in ranked
                ),
            ]
            for category in categories
        ]
        return header, rows

    def summary_table(
        self, k: int = 10, dataset: str | None = None, variant: str | None = None
    ) -> tuple[list[str], list[list[str]]]:
        """Build the compact table: newest run per system at one k, best F1 first.

        Scoping rules are `latest_at_k`'s.

        Returns:
            The header labels and the formatted rows, ranked best F1 first.
        """
        ranked = self.latest_at_k(k, dataset, variant)
        header = ["system", "version", "last run"]
        header += [label for _, label, _ in SUMMARY_COLUMNS]
        header += ["p50 search (s)"]
        rows = []
        for s in ranked:
            cells = [
                str(s.get("system", "?")),
                str(s.get("system_version") or ""),
                run_date(s.get("run_id")),
            ]
            cells += [self._cell(s, key, fmt) for key, _, fmt in SUMMARY_COLUMNS]
            latency = s.get("search_latency", {})
            cells += [f"{latency.get('p50_s', 0):.4f}" if latency else ""]
            rows.append(cells)
        return header, rows

    @staticmethod
    def _cell(summary: dict, key: str, fmt: str) -> str:
        """One comparison cell, saying how much of the cost it accounts for.

        Three states stay apart: a full number is comparable; partial coverage
        is starred, not quietly ranked against complete numbers; unmeasurable
        says "n/a" — a blank would read as a missing datum, not a property.
        """
        coverage = summary.get("usage_coverage", "full")
        if key in summary:
            cell = fmt.format(summary[key])
            if key == "memory_tokens_total" and coverage == "partial":
                return f"{cell}*"
            return cell
        if key == "memory_tokens_total" and coverage == "none":
            return "n/a"
        return ""

    # cost-column footnotes, emitted only when a row actually uses one
    USAGE_FOOTNOTES = {
        "*": "incomplete — the system spends where this harness cannot fully see it",
        "n/a": "not measurable — none of this system's spend is observable here",
    }

    def to_summary_markdown(
        self, k: int = 10, dataset: str | None = None, variant: str | None = None
    ) -> str:
        """Render `summary_table` as a markdown table, with its cost notes."""
        header, rows = self.summary_table(k, dataset, variant)
        table = markdown_table(header, rows)
        try:
            column = header.index("memory tokens")
        except ValueError:
            return table
        notes = self._usage_notes([row[column] for row in rows])
        return table if not notes else table + "\n\n" + notes + "\n"

    @classmethod
    def _usage_notes(cls, cells: list[str]) -> str:
        """Explain the cost marks this table actually used, and only those."""
        used = [
            f"`{mark}` {text}"
            for mark, text in cls.USAGE_FOOTNOTES.items()
            if any(cell == mark or cell.endswith(mark) for cell in cells)
        ]
        return "  \n".join(used)

    def table(self) -> tuple[list[str], list[list[str]]]:
        """Build the full comparison: one row per run, every identity column.

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
