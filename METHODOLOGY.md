# Methodology

How the benchmark is designed and run: design principles, evaluation axes, the harness architecture, datasets, model selection, run scoping, and cost.

## Design principles

1. **Reproducibility first.** Every benchmark run is reproducible from a tagged commit, declared input dataset, and pinned model versions for any LLM calls. Results without reproducibility instructions don't get published.
2. **Fairness over flattery.** Each system is configured according to its docs' recommended setup. We don't tune any system specifically for the benchmark, and we don't tune the benchmark to favor any system.
3. **Multiple axes, no single ranking.** A "best agent memory" system doesn't exist universally — it depends on what you optimize for. Results are reported per-axis, not as an aggregate score.
4. **Transparent about limits.** Where a system can't run a given benchmark (e.g., requires a paid API key, has architectural limitations), we report that explicitly rather than skipping silently.
5. **Public datasets only.** All evaluation data is either created by us and published, or drawn from existing public benchmarks with attribution.

## Evaluation axes

Every run's summary reports these axes (the metric set is `default_metrics()` in `src/amb/metrics.py`):

### Answer accuracy

Did the agent, searching the system, answer correctly? `judge_accuracy` is the share of questions the judge grades correct, with its 95% Wilson interval (`judge_accuracy_ci`), also broken down by the dataset's question category (LoCoMo: single-hop, multi-hop, temporal, open-domain, adversarial; LongMemEval: its `question_type`s). Token-overlap `answer_f1` and `exact_match` are computed too but never headlined.

### Retrieval

Did the agent's searches surface the evidence? Retrieved ids are scored against the dataset's evidence labels as two separate families: session-level `retrieval_precision/recall/f1` (every system can attest a memory's session) and turn-level `turn_precision/recall/f1` (only verbatim-turn stores, like naive). Session-level recall is what the summary and per-category tables rank by; precision mostly measures how sparingly the agent searched, so it is not headlined. `hits_per_query` and `sessions_per_query` show how much the agent looked at per question; see [k is a ceiling, not a quota](#k-is-a-ceiling-not-a-quota).

### k is a ceiling, not a quota

`--k` is the most hits one search may return. It is never padded and never exceeded: a system that finds less returns less, and what came back is what gets scored. The harness does not truncate, so an adapter returning more than k is a bug in the adapter. k caps each *search*, not the question. The agent may search several times, and `hits_per_query` (the comparison table's `hits/q`) counts every hit across those searches, with `sessions_per_query` counting the distinct sessions they cite. Measured on LoCoMo, mem0 2.0.20 at k=10 runs 1.84 searches/question, returns exactly 10 hits on each, and reports 18.42 hits/q. Each `results.jsonl` row records `num_searches`, so hits per search can be recovered from any run.

So compare systems at matched hits/q, not matched k. Recall rises with every hit the agent sees, so at the same k a system whose agent searched more often, or whose searches came back full, can look like it ranks better when the agent has only read more of it. The other direction is just as real. In the direct-mode runs withdrawn in #242 (still in git history), EverOS 1.2.3 returned 1.83, 2.93 and 5.57 hits per search at k = 3, 5 and 10, while every other system returned the full k. RESULTS.md's "scores vs hits returned" charts place each k at its measured hits/q for exactly this comparison.

### Latency

`ingest.total_s` (summed ingestion), plus p50/p95/p99 of per-question `search_latency` (the time spent in all of the agent's searches for that question) and `answer_latency` (the whole agent turn).

### Token spend

The memory system's own spend, `memory_tokens.ingest` and `memory_tokens.search_total`, counted by its spend tracker (`usage_coverage` records how). The question agent's tokens are reported separately per question (`tokens.input_per_question` / `output_per_question`) and never counted as memory spend.

### Not measured yet

Memory footprint (bytes per stored fact), setup complexity (integration SLOC, dependencies, config knobs) and update semantics (what happens when a fact is updated, removed or contradicted) are on the roadmap. No run reports them today.

## Harness design

The repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two kinds of packages, dbt-style:

- **`src/amb`** — the evaluation library only: the normalized conversation schema, dataset loaders (LoCoMo, LongMemEval), metrics, the pydantic-ai answer agent + LLM judge, the `Runner` that executes a benchmark, the `amb` CLI, and a `naive` BM25 baseline. It knows nothing about specific memory systems.
- **`benchmarks/<provider>/`** — one workspace package per integration (like `dbt-postgres` against `dbt-core`). Each implements the base classes from `amb.base`: a `Memory` subclass (`memory.py`) and its specialized `Benchmark` subclass (`benchmark.py`), registered under the `amb.systems` entry-point group so `amb run` invokes the integration's own object. Each also ships its own `pyproject.toml`, a standalone uv `Dockerfile`, and a `docker-compose.yaml` wiring any backing database (Neo4j for Graphiti, the Fraise server, ...) to the benchmark runner.

**A benchmark declares, the runner executes, callbacks measure.** A `Benchmark` says *what* is under test (which `MemorySystem`, its `--param`s, the search tools it exposes to the agent, the callbacks that meter it) and never sees a `RunConfig` — the same split HuggingFace draws between a model and its `Trainer`. `amb.runner.Runner` owns the protocol: sample selection, the turn budget, the question budget, parallelism. It records nothing itself; every phase is bracketed by a hook, and everything a run reports about itself — latencies from `TimingTracker`, token spend from `OpenAIUsageTracker` — is produced by an observer attached to those hooks. So a new axis of measurement is a new callback, never an edit to the loop, and `Runner.core_callback_classes` attaches the ones no run may be missing before any the benchmark declares:

```python
data = Runner(FraiseBenchmark(params={"model": "openai:gpt-5-mini"}),
              RunConfig(dataset="locomo", model="anthropic:claude-haiku-5-5", k=10)).run()
```

Every experiment is the same pipeline — **ingest → qa → evaluation** — measured the way an agent uses memory:

- **Ingest:** each session goes through the adapter's `ingest_session`, i.e. the system's **own extractor** (mem0's fact extraction, graphiti's entity extraction, hindsight's server-side retain, ...), on the model set with `--llm`.
- **QA:** amb's question agent (`--model`, required) searches through the system's *own* search tools (mem0's `search_memory`, hindsight's `recall`, fraise's `query` (raw FQL recalls, written from the language spec in its instructions), ... or the generic `search_memory`) as many times as it likes, then answers. Retrieval is recorded on everything it chose to look at.
- **Evaluation:** the judge (`--judge-model`) gives each answer one binary verdict, correct or not. **Judge accuracy is the headline**, reported with its 95% Wilson interval — a few hundred verdicts per run leave a few points of noise, so systems inside each other's interval are not ranked as different. Session-level retrieval recall is the diagnostic beside it: whether the agent found the evidence at all. LoCoMo's adversarial questions are unanswerable; the judge counts a correct abstention ("I don't know") as correct there, so the verdict stays binary.

There is no retrieval-only mode: earlier releases scored a harness-driven search on retrieval F1, and those results were withdrawn — they measured a search engine, not memory in an agent.

`Runner.run()` does: setup -> ingest sessions (timed) -> per question: the agent searches (each search timed) and answers -> record retrieval against evidence labels -> teardown. Judging is an **evaluation** step, not part of the run: `amb judge <run-dir>` (or `amb run --judge` as a convenience) grades saved predictions, and can re-grade them with a different judge without re-running the system. Every run automatically writes its JSON report (`summary.json`, per-question `results.jsonl`, `ingest.json`) to `runs/<dataset>/<system>/<run-id>/` in the repo (`runs/<dataset>/<variant>/<system>/<run-id>/` when the run named a `--variant` — a different dataset variant is a different experiment, not a different run of the same one, so it gets its own path level) — the Docker environments volume-mount `runs/` so containerized runs land there too. `ComparisonReport` aggregates saved runs into the final cross-system comparison (`amb report`), and `amb.reporting` turns that comparison into the published document: [RESULTS.md](RESULTS.md) is generated, never hand-edited. The [README's results section](README.md#results) is hand-written and embeds the generated charts by path. Each report is a list of sections, and **a section declares the charts it shows and links exactly those** — so `amb plot all` draws precisely the figures the documents reference, and a new dataset, variant, or model generation adds a section (and its own `plots/` directory) by appearing in `runs/`, with no code change.

## Datasets

| Dataset | Source | Variants |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) (GitHub JSON) | default |
| LongMemEval | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) (HuggingFace) | s, m, oracle, xs, xs-shared |

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

`xs` and `xs-shared` are not published files: the loader cuts them from `s` at load time, so they share its labels and need no extra download. They exist to put LongMemEval's question types in a run that ingests in about LoCoMo's time. Every question costs at least its evidence sessions — all 500 questions' evidence is the 940-session `oracle` corpus, 3–4× LoCoMo — so both cuts ask the same 151 questions, drawn from `s` per question type and abstention flag in proportion (seeded, kept in file order).

- **`xs`** — one history per question, holding only its evidence sessions in timestamp order: the control. 151 histories, 293 sessions, 3,346 turns.
- **`xs-shared`** — LoCoMo's shape on the same evidence. The questions are dealt round-robin into 10 synthetic users (`user_0`…`user_9`, 14–15 questions each); a user's history is its questions' evidence sessions plus 5 fillers drawn from their haystacks (seeded by the user id), deduplicated by session id and timestamp-ordered, so each question's distractors are mostly the other questions' evidence. Two questions with the same wording never share a user. The 10 abstention questions (`*_abs`) keep an `xs` history of their own instead, since another question's evidence could answer what they ask about. 20 histories, 343 sessions, 3,807 turns; a user holds 28–38 sessions and 300–434 turns (LoCoMo: 19–32 sessions, ~590 turns).

The gap between `xs` and `xs-shared` is what a shared history costs, on identical evidence. Every question keeps its own `question_date`, and a shared history spans all its questions' dates, so on `xs-shared` 131 of the 141 shared questions see sessions dated after the question is asked (76 of 500 on `s` itself). The answer agent is told the question's date, so it can discount them; the memory system is not.

## Models

A run uses two independent sets of models, and they are worth keeping straight because they measure different things:

- **Ingestion models** — what the *memory system* calls internally to extract and embed facts. This is the system under test, and it dominates run cost.
- **Answering models** — what the *harness* calls to answer benchmark questions and (with `--judge`) grade them: amb's question agent and judge. This is the evaluator, and it should stay fixed while you compare memory systems. The published runs use `anthropic:claude-haiku-5-5` for both (needs `ANTHROPIC_API_KEY`), and as every system's `--llm`, with `openai:text-embedding-3-small` as its `--embedding`.

Every model is chosen on the command line — there are no model environment variables, so a run is fully described by its command:

| Flag | Controls | Default |
| --- | --- | --- |
| `--llm` | memory system's own LLM (extraction) | `openai:gpt-5-mini`; naive has none |
| `--embedding` | memory system's own embedding model | `openai:text-embedding-3-small`; naive has none |
| `--model` | amb's question agent | required — `test` (pydantic-ai's offline model) for smoke runs |
| `--judge-model` | judge (`amb judge` / `--judge`) | follows `--model` |

```bash
uv run amb run --system mem0 --dataset locomo --judge \
  --llm anthropic:claude-haiku-5-5 --embedding openai:text-embedding-3-small \
  --model anthropic:claude-haiku-5-5 --judge-model anthropic:claude-haiku-5-5
```

Every model id has one format, `provider:name` — the pydantic-ai one. `--llm` and `--embedding` are the same flags for every system; each adapter translates the id into its own SDK's form (cognee's `anthropic/...` litellm name, hindsight's server provider/model pair, ...), and a system refuses to start on a provider it cannot run **natively** rather than shimming it or falling back to another model. `--param model=` / `--param embedding_model=` are rejected, so the models have exactly one source. `none` runs a system without that half where it supports it (fraise: no tagger / no embedder). What each system can run today:

| System | `--llm` providers | How anthropic runs | `--embedding` |
| --- | --- | --- | --- |
| mem0, graphiti, langmem | openai, anthropic | the SDK's own anthropic provider | openai |
| cognee | openai, anthropic | its own litellm path | openai |
| hindsight | openai, anthropic | the server's own provider, configured by compose | openai |
| fraise | openai, anthropic | `src/extractor.py`: fraise's `Extractor` interface with the SDK's own prompt and schema (the SDK ships only OpenAI's) | openai |
| memos | openai, anthropic | an `AnthropicLLM` behind MemOS's LLM interface, in the `openai` backend's slot (as the reasoning accommodation already was) | openai |
| everos | openai, anthropic | a client satisfying everalgo's structural `LLMClient` protocol, in EverOS's process-wide LLM slots | openai |
| naive | — | — | — |

Embeddings stay on OpenAI everywhere: Anthropic has no embedding model. fraise, memos and everos have no anthropic provider of their own, so for those three the provider call is adapter code plugged into the system's own extension point — their prompts, parsing and pipelines are unchanged, but it is a seam the other five do not have. In-process anthropic spend is counted by the same tracker as openai's (`messages.create`/`parse`, cache tokens included); amb's own question agent and judge mark their SDK clients so their tokens land on the row, never in a system's cost.

Hindsight extracts inside its server, so its compose stack is configured from the same id (`HINDSIGHT_API_LLM_PROVIDER` / `HINDSIGHT_API_LLM_MODEL`, plus `HINDSIGHT_API_LLM_API_KEY` for anthropic — the workflow derives all three from `LLM`), and the adapter refuses an `--llm` the server is not running. Every run records the system's effective `ingestion_model`/`embedding_model` and any other `--param` overrides in its summary — they are part of the run's comparison identity, so a variant (fraise with an embedder, mem0 on a different extractor) is a new row, never a replacement.

`--param` is the general channel for memory-system settings, and its values are coerced (`true`/`false` to booleans, numbers to numbers). Mem0 also takes `--param reasoning=false`, which restores its normal sampling arguments; it defaults to true because mem0's own reasoning-model detection misses `gpt-5-mini` and sends a `temperature` the model rejects with a 400.

**Run parameters are CLI-only; infrastructure is environment-only.** Nothing is settable both ways, so a benchmark's behaviour never depends on ambient state — the command fully describes the run, and the same command means the same thing on any machine.

- **CLI** — everything that changes what a run measures: models, `--limit` / `--turns` / `--questions`, `--data-dir`, `--runs-dir`, and dataset sources.
- **Environment** — only where things live and how to authenticate: `OPENAI_API_KEY` (and `OPENAI_BASE_URL`), `ANTHROPIC_API_KEY`, `HF_TOKEN`, and the endpoints and storage roots (`QDRANT_HOST` / `QDRANT_PORT`, `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, `FRAISE_BASE_URL`, `HINDSIGHT_BASE_URL`, `LANGMEM_DATABASE_URL`, `COGNEE_ROOT`, `EVEROS_ROOT`). The compose files set the endpoints; locally they default to localhost.

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
| LongMemEval | one question with its own haystack | many (`s` ≈115k tokens, `m` ≈500 sessions); `xs` 1–6 | exactly 1 | turn + session |
| LongMemEval `xs-shared` | a synthetic user's history, or one abstention question's | 28–38 per user | 14–15 per user, 1 per abstention history | turn + session |

## What `--turns N` costs on each dataset

The budget is in turns, but a turn is a different amount of text in each corpus, so the same `N` buys very different input sizes:

| Dataset | Turns per sample | Turns per session | `--turns N` ingests | Boundary behaviour |
| --- | --- | --- | --- | --- |
| LoCoMo | ~590 (5,882 over 10 conversations) | median 20 (range 10–47) | ≈ N × 31 tokens | usually lands on a session boundary |
| LongMemEval | one haystack per question (`s` ≈115k tokens, `m` ≈500 sessions) | short chat sessions | a prefix of the haystack | usually lands on a session boundary |

The LoCoMo row is measured from the cached corpus and the LongMemEval row from its `oracle` variant. One consequence worth planning around: on LoCoMo a budget below ~20 turns gets you a single partial session.

Consequences worth knowing before planning a run:

- **`--questions` is a no-op on LongMemEval**, except on `xs-shared`. Each sample holds exactly one QA pair; use `--limit` to ask fewer questions. On `xs-shared` it caps questions per user as on LoCoMo, and `--limit` counts histories (the 10 users come first, then the 10 abstention histories), not questions.
- **Ingestion dominates cost for LLM-backed systems.** Mem0 makes one extraction LLM call per session, so cost scales with corpus size, not question count — see [Ingestion cost](#ingestion-cost) below. The `naive` baseline is BM25-only (the whole LoCoMo corpus ingests in 0.04 s), so use it to verify plumbing before spending on a real system.
- **A turn budget makes the scores partial, so the run is marked.** Questions whose evidence falls outside the budget are skipped rather than scored — they are unanswerable by construction, and scoring them would blame the memory system for data it never received. The count lands in the report as `ingest.questions_dropped`. Truncated runs record `max_turns` and form a separate identity in `amb report --latest`, so a smoke run can never displace a real one.
- **The two flags compose in that order: filter, then cap.** `--questions N` gives you N *answerable* questions, not N candidates of which some are dead. On one LoCoMo conversation, `--turns 100` leaves 59 of 199 questions answerable; adding `--questions 5` asks 5 of those 59.
- **Runs can be scaled out and amortized.** `--workers N` runs conversations in parallel (they are independent; sessions within each stay ordered — verified to produce byte-identical scores to a serial run). `--seed N` draws `--limit` conversations at random but reproducibly, and the seed is recorded in the summary and the comparison identity. `--keep` skips teardown so the store retains this run's memories; a later `--reuse` run skips ingestion entirely and queries the store as-is — the report charges `ingest_s: 0` and marks each sample `reused`. **`--reuse` trusts you that the store matches the scope**: run it with the same dataset/limit/turns as the `--keep` run that populated it.
- **Retrieval is scored at every level the labels and the hits both support**, as two metric families that never share a name: session-level `retrieval_*`, which every system can attest, and turn-level `turn_*`, which only verbatim-turn stores (naive) can. Comparisons use only the session-level columns. Hits with no provenance at any labelled level are excluded with a warning rather than scored 0.
- **The filter trusts the dataset's evidence labels.** A question is kept when every evidence turn it names was ingested (or, for session-level labels, when the whole evidence session was). Where labels are incomplete — a multi-hop question needing context the labels don't list — a kept question can still be harder than it would be on the full corpus.

## Ingestion cost

Ingestion is serial within a sample and, for LLM-backed systems, is the dominant cost of a run. Measured with Mem0 (`gpt-5-mini` extraction, Qdrant; the published runs now extract with `claude-haiku-5-5`, which has not been timed) on one LoCoMo conversation — 19 sessions / 419 turns in **17.7 min**, one extraction LLM call per session, 269k LLM tokens:

| Dataset | Corpus | Mem0 ingestion | `naive` ingestion |
| --- | --- | --- | --- |
| LoCoMo | 10 conversations, 272 sessions, 5,882 turns | ~4 h | 0.04 s |
| LongMemEval `oracle` | 500 instances, 948 sessions (evidence only) | ~12 h | seconds |
| LongMemEval `s` | 500 instances x ~115k tokens | ~7 weeks | seconds |
| LongMemEval `m` | 500 instances x ~500 sessions | ~5 months | seconds |
| LongMemEval `xs` | 151 histories, 293 sessions, 3,346 turns | ~3.5 h | seconds |
| LongMemEval `xs-shared` | 20 histories, 343 sessions, 3,807 turns | ~4 h | seconds |

The LoCoMo and `oracle` rows are measured end to end — `oracle` from the full 500-instance runs formerly in `runs/longmemeval/oracle/` (withdrawn with the retrieval-only results in #242, which did not change ingestion; still in git history), where Mem0 sums 11.2–11.9 h of ingestion across its four k values. The `s` and `m` rows extrapolate LoCoMo's throughput (~13.6 source tokens/s, ~2.5 s/turn) by corpus size, and their instance counts come from published sizes rather than a downloaded corpus. Treat those two as orders of magnitude, not schedules. The `xs` rows are extrapolated too, from the measured `oracle` row: their corpus sizes are measured from the cut, and `oracle`'s evidence sessions are those same ones (13.1M characters), so scaling `oracle`'s Mem0 time by sessions or by characters gives 31–37 h for `xs` and 15–16 h for `xs-shared`.

Every figure above is *summed* ingestion — the total across all conversations, which is what the reports record and which does not shrink with `--workers`. Wall clock does: `--workers 3` measures ~2.3x on LoCoMo (2.99 h summed, 1.28 h elapsed), the parallel speedup being sublinear because conversations differ in length and the run waits on the longest. Summed across `oracle`, Graphiti measures 4.1 h, Fraise 10.6 h and Mem0 11.8 h, which at that speedup puts Fraise near 4.6 h and Mem0 near 5.1 h of wall clock against CI's six-hour ceiling: past roughly 40 s/session there is no headroom left.

The practical consequence: **LoCoMo, LongMemEval `oracle`, `xs` and `xs-shared` are whole-dataset runs today; `s` and `m` are not.** For those, scope the run with `--limit` and `--turns` and report what you scoped, or expect a multi-day job. Each LongMemEval instance carries its own haystack, so its cost is per question — halving `--limit` halves the run.

### Persistent stores

Each database-backed integration keeps its data in a named Docker volume (`qdrant-storage` for mem0, `neo4j-data`/`neo4j-logs` for graphiti, `memos-neo4j-data`/`memos-neo4j-logs`, and `cognee-pgdata`, `hindsight-pgdata`, `langmem-pgdata`), so a store survives `docker compose ... down` and a crashed or interrupted run does not throw away the ingestion you already paid for. To start from an empty store:

```bash
docker compose -f benchmarks/mem0/docker-compose.yaml down -v   # -v drops the volume
```

Note that persistence alone does not let a second run skip ingestion: each sample's `teardown()` deletes that conversation's memories, so a fresh run re-ingests. To reuse a populated store, opt out explicitly: `--keep` skips teardown, and a later `--reuse` run skips ingestion and queries the store as-is.
