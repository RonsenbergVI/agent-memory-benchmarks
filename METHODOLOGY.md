# Methodology

How the benchmark is designed and run: design principles, evaluation axes, the harness architecture, datasets, model selection, run scoping, and cost.

## Design principles

1. **Reproducibility first.** Every benchmark run is reproducible from a tagged commit, declared input dataset, and pinned model versions for any LLM calls. Results without reproducibility instructions don't get published.
2. **Fairness over flattery.** Each system is configured according to its docs' recommended setup. We don't tune any system specifically for the benchmark, and we don't tune the benchmark to favor any system.
3. **Multiple axes, no single ranking.** A "best agent memory" system doesn't exist universally — it depends on what you optimize for. Results are reported per-axis, not as an aggregate score.
4. **Transparent about limits.** Where a system can't run a given benchmark (e.g., requires a paid API key, has architectural limitations), we report that explicitly rather than skipping silently.
5. **Public datasets only.** All evaluation data is either created by us and published, or drawn from existing public benchmarks with attribution.

## Evaluation axes

The benchmark measures systems across the following axes:

### Recall accuracy

Does the system surface the right facts when asked? Measured against ground-truth labels on multi-turn conversations where the agent must recall earlier-mentioned facts.

- **Single-hop recall**: agent recalls a fact mentioned earlier in the same conversation.
- **Multi-hop recall**: agent recalls a fact connected via relationships to the query.
- **Temporal recall**: agent recalls the most recent fact when multiple versions exist.
- **Cross-session recall**: agent recalls facts from prior sessions in a new session. Metric: F1 on retrieved fact set against ground truth.

### Latency

End-to-end query latency including any LLM calls the memory system makes during recall.

Metric: p50, p95, p99 over fixed query workloads.

### Token efficiency

How many input + output tokens does the agent's interaction with the memory system consume per useful recall? Lower is better.

Metric: average tokens per useful recall, separated by input and output.

### Memory footprint

RAM per stored fact (where measurable) and total memory growth over a fixed conversation length.

Metric: bytes per fact, total bytes after N facts.

### Setup complexity

How many lines of glue code does the agent integration require? How many external dependencies? How many configuration knobs to get a working setup?

Metric: SLOC of integration code, dependency count, required config fields.

### Update semantics

What does the system do when an existing fact is updated, removed, or contradicted?

Reported qualitatively as a feature matrix.

## Test corpora

The benchmark uses two corpora:

- LoCoMo
- LongMemEval

each with different characteristics:

1. **Conversational** (5,000 facts across 200 sessions) — synthetic conversations between an agent and a user, with ground-truth recall queries embedded at varying time offsets.
2. **Knowledge stream** (10,000 facts) — parsed knowledge from a small domain (e.g., Wikipedia subset), with structured queries that require multi-hop reasoning.
3. **Long-horizon** (1,000 facts over 100 sessions) — extreme cross-session recall test, where facts mentioned weeks of conversation ago must be surfaced in current context. Corpora and their generation scripts are published in `/data/`.

## Harness design

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two kinds of packages, dbt-style:

- **`src/amb`** — the evaluation library only: the normalized conversation schema, dataset loaders (LoCoMo, LongMemEval), metrics, the pydantic-ai answer agent + LLM judge, the `Runner` that executes a benchmark, the `amb` CLI, and a `naive` BM25 baseline. It knows nothing about specific memory systems.
- **`benchmarks/<provider>/`** — one workspace package per integration (like `dbt-postgres` against `dbt-core`). Each implements the base classes from `amb.base`: a `Memory` subclass (`memory.py`) and its specialized `Benchmark` subclass (`benchmark.py`), registered under the `amb.systems` entry-point group so `amb run` invokes the integration's own object. Each also ships its own `pyproject.toml`, a standalone uv `Dockerfile`, and a `docker-compose.yaml` wiring any backing database (Neo4j for Graphiti, the Letta server, ...) to the benchmark runner.

**A benchmark declares, the runner executes, callbacks measure.** A `Benchmark` says *what* is under test (which `MemorySystem`, its `--param`s, the tool surface it exposes in agentic mode, the callbacks that meter it) and never sees a `RunConfig` — the same split HuggingFace draws between a model and its `Trainer`. `amb.runner.Runner` owns the protocol: sample selection, the turn budget, parallelism, direct vs agentic. It records nothing itself; every phase is bracketed by a hook, and everything a run reports about itself — latencies from `TimingTracker`, token spend from `OpenAIUsageTracker` — is produced by an observer attached to those hooks. So a new axis of measurement is a new callback, never an edit to the loop, and `Runner.core_callback_classes` attaches the ones no run may be missing before any the benchmark declares:

```python
data = Runner(FraiseBenchmark(params={"model": "gpt-5-mini"}),
              RunConfig(dataset="locomo", k=10)).run()
```

Every experiment is the same pipeline — **ingest → qa → evaluation** — and `--mode` picks who drives the memory calls:

- **`direct`** (default): the harness ingests via the adapter and searches each question verbatim. With `--model`, the qa phase also answers from the retrieved context; without one it stops at retrieval (free, no LLM).
- **`agentic`**: a model drives both sides through the system's *own* tools — it decides what to `remember`/`add`/insert at ingestion and how to search at question time. `--model` is mandatory; an agentic run never falls back to a default model or to direct behavior.

`Runner.run()` does: setup -> ingest sessions (timed) -> per question: search (timed) -> record retrieval against evidence labels -> optionally answer with an LLM -> teardown. Judging is an **evaluation** step, not part of the run: `amb judge <run-dir>` (or `amb run --judge` as a convenience) grades saved predictions, and can re-grade them with a different judge without re-running the system. Every run automatically writes its JSON report (`summary.json`, per-question `results.jsonl`, `ingest.json`) to `runs/<dataset>/<system>/<mode>/<run-id>/` in the repo (`runs/<dataset>/<variant>/<system>/<mode>/<run-id>/` when the run named a `--variant` — a different dataset variant is a different experiment, not a different run of the same one, so it gets its own path level) — the Docker environments volume-mount `runs/` so containerized runs land there too. `ComparisonReport` aggregates saved runs into the final cross-system comparison (`amb report`), and `amb.reporting` turns that comparison into the published documents: [RESULTS.md](RESULTS.md) and the [README's results section](README.md#results) are generated, never hand-edited. Each report is a list of sections, and **a section declares the charts it shows and links exactly those** — so `amb plot all` draws precisely the figures the documents reference, and a new dataset, variant, or model generation adds a section (and its own `plots/` directory) by appearing in `runs/`, with no code change.

## Datasets

| Dataset | Source | Variants |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) (GitHub JSON) | default |
| LongMemEval | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (HuggingFace) | s, m, oracle |

All are normalized into one schema (`Conversation` -> `Session` -> `Turn`, plus `QAPair` with evidence labels), so integrations only ever see one shape of data.

Every loader's job is to map its source's idiosyncratic shape onto that schema. The three sources differ enough that it is worth seeing them side by side.

### LoCoMo

One record per conversation. Sessions are **numbered keys** on a single object rather than a list, with their timestamp in a parallel `session_N_date_time` key. Turns carry a `dia_id` that the QA pairs reference as evidence.

```jsonc
{
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline", "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      { "speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you! How have you been?" }
    ]
    // session_2, session_2_date_time, ... up to session_32
  },
  "qa": [
    { "question": "When did Caroline go to the LGBTQ support group?",
      "answer": "7 May 2023", "evidence": ["D1:3"], "category": 2 }
  ]
}
```

Normalized: one `BenchmarkSample` per conversation, `dia_id` becomes `Turn.turn_id`, `evidence` becomes `QAPair.evidence_turn_ids`, and the numeric `category` maps to a name. **Turn-level evidence**, which is what makes LoCoMo the most precise dataset for retrieval scoring.

### LongMemEval

One record per **question**, each carrying its own haystack of sessions. Sessions are a list of message lists, with ids and dates in parallel arrays, and the sessions containing the answer called out separately.

```jsonc
{
  "question_id": "gpt4_2655b836",
  "question": "What was the first issue I had with my new car after its first service?",
  "answer": "GPS system not functioning correctly",
  "question_type": "temporal-reasoning",
  "question_date": "2023/04/10 (Mon) 23:07",
  "haystack_session_ids": ["answer_4be1b6b4_2", "answer_4be1b6b4_3"],
  "haystack_dates":       ["2023/04/10 (Mon) 17:50", "..."],
  "answer_session_ids":   ["answer_4be1b6b4_1", "answer_4be1b6b4_2", "answer_4be1b6b4_3"],
  "haystack_sessions": [
    [ { "role": "user", "content": "I'm thinking of getting my car detailed soon...",
        "has_answer": true } ]
  ]
}
```

Normalized: one `BenchmarkSample` per question, so `qa` always has exactly one entry — this is why `--questions` does nothing here. Turn ids are synthesised as `<session_id>:<index>`; `answer_session_ids` becomes `evidence_session_ids`, and any turn flagged `has_answer` also contributes `evidence_turn_ids`. The `oracle` variant ships only the evidence sessions (3 above), while `s` and `m` bury them in a much larger haystack.

## Models

A run uses two independent sets of models, and they are worth keeping straight because they measure different things:

- **Ingestion models** — what the *memory system* calls internally to extract and embed facts. This is the system under test, and it dominates run cost.
- **Answering models** — what the *harness* calls to answer benchmark questions and (with `--judge`) grade them. This is the evaluator, and it should stay fixed while you compare memory systems.

Every model is chosen on the command line — there are no model environment variables, so a run is fully described by its command:

| Flag | Controls | Default |
| --- | --- | --- |
| `--param model=...` | memory system's extraction LLM | mem0 `gpt-5-mini`; graphiti keeps its own; fraise/naive none |
| `--param embedding_model=...` | memory system's embedding model | mem0 `text-embedding-3-small`; graphiti keeps its own; fraise/naive none |
| `--model` | harness answer agent | none — direct runs skip answering without it; agentic runs refuse to start |
| `--judge-model` | judge (`amb judge` / `--judge`) | follows `--model` |

```bash
uv run amb run --system mem0 --dataset locomo --judge \
  --param model=gpt-4.1-mini --param embedding_model=text-embedding-3-small \
  --model openai:gpt-5-mini --judge-model openai:gpt-5-mini
```

Note the id formats differ: answering models are pydantic-ai ids (`openai:gpt-5-mini`), while ingestion models are raw provider ids (`gpt-5-mini`) because each memory SDK talks to its provider directly. Every run records the system's effective `ingestion_model`/`embedding_model` and any other `--param` overrides in its summary — they are part of the run's comparison identity, so a variant (fraise with an embedder, mem0 on a different extractor) is a new row, never a replacement.

`--param` is the general channel for memory-system settings, and its values are coerced (`true`/`false` to booleans, numbers to numbers). Mem0 also takes `--param reasoning=false`, which restores its normal sampling arguments; it defaults to true because mem0's own reasoning-model detection misses `gpt-5-mini` and sends a `temperature` the model rejects with a 400.

**Run parameters are CLI-only; infrastructure is environment-only.** Nothing is settable both ways, so a benchmark's behaviour never depends on ambient state — the command fully describes the run, and the same command means the same thing on any machine.

- **CLI** — everything that changes what a run measures: models, `--limit` / `--turns` / `--questions`, `--data-dir`, `--report-dir`, and dataset sources.
- **Environment** — only where things live and how to authenticate: `OPENAI_API_KEY`, `HF_TOKEN`, and the database endpoints (`QDRANT_HOST` / `QDRANT_PORT`, `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, `LETTA_BASE_URL`, `FRAISE_BASE_URL`). The compose files set the endpoints; locally they default to localhost.

`src/amb` reads no environment variables at all; only the `benchmarks/*` adapters do, and only for their endpoints, each with a localhost default. One repo-root `.env` can hold every integration's variables at once — the compose files pass it through, and unrelated variables are ignored.

## Run scope: what `--limit` and `--questions` control

A run is `for sample in samples: ingest every session, then ask every question`. The two flags scope those loops:

- **`--limit N`** caps how many **samples** are loaded.
- **`--turns N`** caps how many turns of each sample are **ingested** — the corpus-size knob, and the only one that makes a single sample cheap. Sessions are taken in chronological order until the budget runs out; the session that straddles the boundary is truncated, so what is ingested is always a valid prefix.
- **`--questions N`** caps questions **per sample** — workload, not corpus size. It applies after ingestion, so it never makes ingestion cheaper.

"One sample" means something different in each dataset, which decides what those flags buy you:

| Dataset | One sample is | Sessions per sample | Questions per sample | Evidence labels |
| --- | --- | --- | --- | --- |
| LoCoMo | a conversation | 19–32 | 105–199 | turn + session |
| LongMemEval | one question with its own haystack | many (`s` ≈115k tokens, `m` ≈500 sessions) | exactly 1 | turn + session |

## What `--turns N` costs on each dataset

The budget is in turns, but a turn is a different amount of text in each corpus, so the same `N` buys very different input sizes:

| Dataset | Turns per sample | Turns per session | `--turns N` ingests | Boundary behaviour |
| --- | --- | --- | --- | --- |
| LoCoMo | ~590 (5,882 over 10 conversations) | median 20 (range 10–47) | ≈ N × 31 tokens | usually lands on a session boundary |
| LongMemEval | one haystack per question (`s` ≈115k tokens, `m` ≈500 sessions) | short chat sessions | a prefix of the haystack | usually lands on a session boundary |

The LoCoMo row is measured from the cached corpus and the LongMemEval row from its `oracle` variant. One consequence worth planning around: on LoCoMo a budget below ~20 turns gets you a single partial session.

Consequences worth knowing before planning a run:

- **`--questions` is a no-op on LongMemEval.** Each sample holds exactly one QA pair; use `--limit` to ask fewer questions.
- **Ingestion dominates cost for LLM-backed systems.** Mem0 makes one extraction LLM call per session, so cost scales with corpus size, not question count — see [Ingestion cost](#ingestion-cost) below. The `naive` baseline is BM25-only (the whole LoCoMo corpus ingests in 0.04 s), so use it to verify plumbing before spending on a real system.
- **A turn budget makes the scores partial, so the run is marked.** Questions whose evidence falls outside the budget are skipped rather than scored — they are unanswerable by construction, and scoring them would blame the memory system for data it never received. The count lands in the report as `ingest.questions_dropped`. Truncated runs record `max_turns` and form a separate identity in `amb report --latest`, so a smoke run can never displace a real one.
- **The two flags compose in that order: filter, then cap.** `--questions N` gives you N *answerable* questions, not N candidates of which some are dead. On one LoCoMo conversation, `--turns 100` leaves 59 of 199 questions answerable; adding `--questions 5` asks 5 of those 59.
- **Runs can be scaled out and amortized.** `--workers N` runs conversations in parallel (they are independent; sessions within each stay ordered — verified to produce byte-identical scores to a serial run). `--seed N` draws `--limit` conversations at random but reproducibly, and the seed is recorded in the summary and the comparison identity. `--keep` skips teardown so the store retains this run's memories; a later `--reuse` run skips ingestion entirely and queries the store as-is — the report charges `ingest_s: 0` and marks each sample `reused`. **`--reuse` trusts you that the store matches the scope**: run it with the same dataset/limit/turns as the `--keep` run that populated it.
- **Retrieval is scored at the level each system can attest.** Systems citing turns (naive, letta) score turn-level; systems citing sessions (mem0, graphiti) score session-level. The level is recorded per row and summarised as `retrieval_levels` — never compare recall across levels.
- **The filter trusts the dataset's evidence labels.** A question is kept when every evidence turn it names was ingested (or, for session-level labels, when the whole evidence session was). Where labels are incomplete — a multi-hop question needing context the labels don't list — a kept question can still be harder than it would be on the full corpus.

## Ingestion cost

Ingestion is serial and, for LLM-backed systems, is the dominant cost of a run. Measured with Mem0 (`gpt-5-mini` extraction, Qdrant) on one LoCoMo conversation — 19 sessions / 419 turns in **17.7 min**, one extraction LLM call per session, 269k LLM tokens:

| Dataset | Corpus | Mem0 ingestion | `naive` ingestion |
| --- | --- | --- | --- |
| LoCoMo | 10 conversations, 272 sessions, 5,882 turns | ~4 h | 0.04 s |
| LongMemEval `oracle` | 500 instances, 948 sessions (evidence only) | ~12 h | seconds |
| LongMemEval `s` | 500 instances x ~115k tokens | ~7 weeks | seconds |
| LongMemEval `m` | 500 instances x ~500 sessions | ~5 months | seconds |

The LoCoMo and `oracle` rows are measured end to end — `oracle` from the full 500-instance runs in `runs/longmemeval/oracle/`, where Mem0 sums 11.2–11.9 h of ingestion across its four k values. The `s` and `m` rows extrapolate LoCoMo's throughput (~13.6 source tokens/s, ~2.5 s/turn) by corpus size, and their instance counts come from published sizes rather than a downloaded corpus. Treat those two as orders of magnitude, not schedules.

Every figure above is *summed* ingestion — the total across all conversations, which is what the reports record and which does not shrink with `--workers`. Wall clock does: `--workers 3` measures ~2.3x on LoCoMo (2.99 h summed, 1.28 h elapsed), the parallel speedup being sublinear because conversations differ in length and the run waits on the longest. Summed across `oracle` the four benchmarked systems span an order of magnitude — Letta 1.0 h, Graphiti 4.1 h, Fraise 10.6 h, Mem0 11.8 h — which at that speedup puts Fraise near 4.6 h and Mem0 near 5.1 h of wall clock against CI's six-hour ceiling. That thin headroom is what the [dataset tiers](README.md#dataset-tiers) rule protects: past roughly 40 s/session there is none left.

The practical consequence: **LoCoMo and LongMemEval `oracle` are whole-dataset runs today; `s` and `m` are not.** For those, scope the run with `--limit` and `--turns` and report what you scoped, or expect a multi-day job. Each LongMemEval instance carries its own haystack, so its cost is per question — halving `--limit` halves the run.

### Persistent stores

Each integration's database keeps its data in a named Docker volume (`qdrant-storage`, `neo4j-data`/`neo4j-logs`, `letta-pgdata`), so a store survives `docker compose ... down` and a crashed or interrupted run does not throw away the ingestion you already paid for. To start from an empty store:

```bash
docker compose -f benchmarks/mem0/docker-compose.yaml down -v   # -v drops the volume
```

Note that persistence alone does not let a second run skip ingestion: each sample's `teardown()` deletes that conversation's memories, so a fresh run re-ingests. To reuse a populated store, opt out explicitly: `--keep` skips teardown, and a later `--reuse` run skips ingestion and queries the store as-is.
