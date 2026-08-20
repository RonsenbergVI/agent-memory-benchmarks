# AGENTS.md

Rules for working in this repo. They exist because breaking them has cost us real debugging time — each one is a decision already made, not a suggestion.

## Commands

Run these from the repo root. `make help` lists everything.

| Command | When |
| --- | --- |
| `make test` | After any code change. Fast (~2 s), no API calls. |
| `make check` | **Before committing.** Runs every pre-commit hook on all files. |
| `make smoke` | After changing the run loop, metrics, or reporting — runs ingest → retrieval → report on the `naive` baseline. Answering and judging are skipped (no `--model`), which is why it makes no API calls. |
| `make docs` | After changing reporting — regenerates the charts and the generated documents (`amb plot all` + `amb report --latest --output RESULTS.md --summary README.md`). |
| `make sync` | After changing dependencies or pulling. |
| `make hooks` | Once per clone, to install the git pre-commit hook. |

`make check` is the gate: ruff lint and format, the `ty` type check, hadolint on the Dockerfiles (the `hadolint-docker` variant — it needs a running Docker daemon), the `insert-license` stamper, and the hygiene hooks — including `detect-private-key` and `debug-statements`. Hooks rewrite files (ruff `--fix`, end-of-file and whitespace fixers, and the license stamper, which writes the MIT header from LICENSE onto any Python file missing it), so a first run can fail having already fixed things; re-run it and read the diff. pre-commit itself is not a dev dependency: `uv run pre-commit` falls through to whatever is on PATH, and Homebrew's 2.17.0 aborts on the pinned hooks with an InvalidManifestError — use a modern pre-commit (CI runs `uvx pre-commit`).

`make test` and `make check` are separate on purpose: the hooks do not run the test suite, so passing `check` alone proves nothing about behaviour. Run both — knowing that today the suite covers only the `naive` baseline (`tests/memory_test.py`); every other file under `tests/` is a license-header stub, so a green run says nothing about datasets, metrics, the CLI, or reporting. Fill in the matching stub when you touch those areas. Name new test files `*_test.py` (pytest's default conventions — there is no `python_files` override, so other names are silently not collected), and leave `--import-mode=importlib` in `addopts` alone: `tests/` has no `__init__.py` and reuses basenames across subdirectories, which the default import mode cannot collect. `make typecheck` runs `ty` alone, much faster than the full gate when iterating — but the fast targets (`lint`, `format`, `typecheck`) run unpinned `uvx` tools while the gate pins its ruff and ty versions, so when they disagree, the gate is the arbiter. `ty` also checks `tests/`; the ruff `ANN`/`D` exemption there does not exempt type errors.

Lint policy lives in `[tool.ruff.lint]` in `pyproject.toml` and is deliberately strict: google-convention docstrings (`D`, `DOC`) and full type annotations (`ANN`) on everything public, plus `N`, `UP`, `PT` and import sorting. `tests/**` is exempt from `ANN` and `D` — tests still need `PT` compliance (no composite `assert a and b`, narrow `pytest.raises`).

Write the docstring and the annotations as you add code. They are not optional extras here, and retrofitting them across a package is far more work than writing them once.

Anything that costs money or takes real time — a Mem0, Graphiti, Letta, or fraise run — is **not** in the Makefile. Those go through each integration's own `docker compose` file, by hand, as documented in the README's Quickstart. The run mechanics themselves — the model flags and their defaults, `--limit` / `--turns` / `--questions` semantics, ingestion cost, `--keep`/`--reuse` — are documented in METHODOLOGY.md, not the README; a change to run mechanics updates METHODOLOGY.md, keeping its measured-vs-extrapolated markings.

## CI and releases

`.github/workflows/ci.yml` runs on every pull request and every push to main: one job runs `pre-commit run --all-files` (the same hooks as `make check`, so local and CI cannot drift), another runs `uv sync --locked` and the test suite. `--locked` fails on a stale `uv.lock`, so **commit the lockfile whenever dependencies change.** Integration benchmarks never run in PR CI — they need databases and paid API keys.

They run in the per-framework benchmark workflows instead. `.github/workflows/fraise.yml` runs the fraise benchmark on every `v*` tag push (and on manual dispatch): locomo and longmemeval matrix jobs sweep k ∈ {1, 3, 5, 10} through the integration's own compose file with the `OPENAI_API_KEY` secret and upload `report/` as artifacts; the `publish` job then checks out main, merges the artifacts, runs `amb plot all` and `amb report --latest --k $SUMMARY_K --output RESULTS.md --summary README.md`, and commits `report/`, `plots/`, RESULTS.md, and README.md to main as github-actions[bot]. That commit is the machinery behind "Generated documents" below — and the reason hand edits to those files don't survive a tag. All benchmark runs share one `benchmark` concurrency group, so a second trigger queues rather than interleaving artifacts. A new integration gets its own copy of this workflow (see CONTRIBUTING.md).

`.github/workflows/release-please.yml` keeps a release PR open against `main`, accumulating changes into `CHANGELOG.md` and a version bump; merging that PR tags the release. It bumps the root `amb` version and, via `extra-files`, the four `benchmarks/*/pyproject.toml` versions in lockstep — which is why those version lines carry an `# x-release-please-version` comment. Don't remove it. The config and manifest live under `.github/` — the workflow passes `config-file`/`manifest-file` explicitly, so editing a root-level copy does nothing.

**Commit messages must be [conventional commits](https://www.conventionalcommits.org)** (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, ...). Release-please derives the version bump from them, and ignores commits it cannot parse — a release of only unparseable commits produces no release PR at all. Use `feat!:` or a `BREAKING CHANGE:` footer for anything that breaks the CLI, the entry-point contract, or a data contract.

## Configuration: one source per value

**Run parameters are CLI-only. Infrastructure is environment-only. Nothing is settable both ways.** A value with two sources means ambient state can silently change what a benchmark measures — a recipe for irreproducible results.

- **CLI** — anything that changes what a run measures: models (`--model`, `--judge-model`, `--param model=`), scope (`--limit`, `--turns`, `--questions`), paths (`--data-dir`, `--report-dir`), dataset sources.
- **Environment** — only where things live and how to authenticate: `OPENAI_API_KEY`, `HF_TOKEN`, and database endpoints (`QDRANT_HOST`, `NEO4J_URI`, `LETTA_BASE_URL`, `FRAISE_BASE_URL`, ...) with localhost defaults.

`src/amb` reads **zero** environment variables. Only the `benchmarks/*` adapters read them, and only for endpoints. Never add an `os.environ.get(...)` fallback to something the CLI already sets, and never put a secret on a command line.

## Architecture

- **`src/amb` is evaluation only** — schema, dataset loaders, metrics, the pydantic-ai answer agent and judge, reporting, the `amb` CLI, and the `naive` BM25 baseline. It knows nothing about any specific memory system.
- **Integrations live in `benchmarks/<provider>/`** as uv workspace packages, dbt-adapter style: their own `pyproject.toml` depending on `amb` via the workspace source, an entry point under the `amb.systems` group, a `Memory` subclass in `memory.py`, its `Benchmark` subclass in `benchmark.py`, and the agentic-mode tool surface in `toolset.py` — `SearchToolset`/`IngestToolset` subclasses exposing the system's *native* verbs (letta's `archival_memory_search`, mem0's `add_memory`, fraise's `recall`/`remember`), wired via `search_toolset_class`/`ingest_toolset_class` on the Benchmark. The entry point resolves to the **Benchmark**, not the Memory — otherwise declared callbacks and toolsets never attach.
- **The inner module is always `src`**, in every integration, because the SDKs own the pretty import names (`mem0`, `letta`). Consequence: integrations cannot be co-installed; use `uv sync --package <name>` and never `--all-packages`.
- **The library never orchestrates Docker.** No subprocess calls to docker/compose, no delegation from `amb run`. The same command runs locally and inside a container; users invoke `docker compose` themselves.
- **No implementation in `__init__.py`** — real modules only; `__init__` holds re-exports and, at most, the package's registry wiring (`datasets.LOADERS`/`get_loader`, `cli.commands.COMMANDS`).
- **No root or base Dockerfile.** Each integration ships its own standalone uv Dockerfile plus a `docker-compose.yaml` wiring its database. The Dockerfiles pin the uv binary itself (a `COPY --from=ghcr.io/astral-sh/uv:<version>`) because the floating base-image tag lags the repo's required uv — don't "simplify" that COPY away.

The step-by-step contract for adding an integration or a dataset — entry-point shape, the provenance and spend contracts, toolset rules, `uv lock` after adding the package, the copied benchmark workflow — is CONTRIBUTING.md.

## The data/metrics pipeline

Three stages, decoupled in this order — do not collapse them:

1. **`Runner.run()` returns raw data only** (`Run`): per-question rows and per-sample ingest stats, as pydantic models. It computes **no scores** and writes **no files**. `base/benchmark.py` must not import `reporting`.
2. **Metrics are Keras-style stateful objects** (`metrics.py`): every observation record is fed to every metric via `update_state(record)`; each metric grabs the fields that apply to it, accumulates, and computes the final value in `result()`. `reset_state()` starts a new scope. Scoring functions live *inside* their metric class, not beside it.
3. **`RunReport` consumes the data with a metric set** and owns persistence. The metric set is an input, so saved raw data can be re-scored later without re-running the benchmark.

Dots in a metric name nest its result (`ingest.total_s` → `summary["ingest"]["total_s"]`), so the summary's shape is declared by the metric list, not by assembly code.

Two more moving parts sit around that core:

- **`--mode agentic` swaps the driver, not the pipeline.** A pydantic-ai agent (`src/amb/agent/`) ingests and searches through the Benchmark's declared toolsets; `Runner` refuses agentic mode unless both toolset classes and `--model` are set — it never falls back to direct behaviour. Toolset calls report through the same callback hooks (`record`/`record_write`), so retrieval and latency stay scoreable, and provenance ids are deliberately kept out of tool results — they are the ground-truth labels retrieval is scored against.
- **Judging is an evaluation step, not part of the run.** `RunReport.judge(model)` grades saved predictions and writes `judge_correct`/`judge_reasoning` onto the rows — deliberately outside the run loop, so a saved run can be re-judged with a different model without re-running the memory system. `amb judge <run-dir>` does this; `amb run --judge` is the convenience form.

**Callbacks are data producers**, not scorers: they observe what the harness can't (SDK traffic, server state) and write extra fields into the observation records, which metrics then pick up. Attach them declaratively via `callback_classes`. `Runner.core_callback_classes` (`TimingTracker`) is always prepended, so no benchmark override can drop the latency measurements; `OpenAIUsageTracker` is the default spend tracker (patches the openai SDK with thread-local counters), and `TiktokenUsageTracker` is the opt-in for server-backed systems that report spend through `Memory.usage_counters()` (letta today). Don't re-implement timing or token counting in an integration. A new callback must pick a concurrency strategy: with `--workers`, each sample's whole lifecycle runs on one thread (thread-local counters work), but agentic-mode events arrive from pydantic-ai's own worker thread (key state by sample, as `TimingTracker` does).

## Generated documents

**RESULTS.md and the README's results section are outputs, not files.** They are generated by `amb report` from `src/amb/reporting/report.py`; anything typed into RESULTS.md by hand is gone on the next run. Prose that belongs in the results document belongs in the report definition. Until the first `v*` tag they are placeholder stubs; the benchmark workflows' `publish` job fills them (see CI and releases), and `make docs` runs the same two commands locally.

The README's generated region is delimited by `<!-- amb:summary -->` / `<!-- /amb:summary -->`. `amb report --summary` replaces only what sits between the markers — and a missing or renamed marker does not fail, it **appends a fresh `## Results` region at the end of the file**, i.e. a duplicate section. Never delete or move the markers when hand-editing the README.

One rule makes the chart/document coupling safe, and it is the reason the module exists:

- **A section declares the charts it shows, and renders its figures from that same list.** `Section.charts()` and `Section.blocks()` are two consumers of one list, never two lists kept in agreement. `amb report` writes the documents; `amb plot all` draws the charts those same documents declare. This replaced a hand-maintained plots section that had already drifted — it linked `k=1/3/5` images that CI never generated.
- **A chart with no data is dropped before it is referenced** (`Chart.has_data()`), so a document cannot link a missing image. The count of dropped charts is reported, never silent.
- **Sections come from the data.** `group_runs` splits runs by `(dataset, variant)` — the axes along which runs are different *experiments* rather than repeats — so a new dataset or variant appears in `report/` and gets its own section and its own `plots/<dataset>[/<variant>]/` directory with no code change. Extend `group_by` (e.g. with `ingestion_model`) the day a second model generation runs, and every group splits, directories included.
- **A group spanning several modes is refused**, not blended: a direct and an agentic run of one system are different subjects. Promote `mode` into `group_by` when both are worth publishing side by side.
- **Pinned precision drops charts by design.** When every run in a group reports `retrieval_precision` of exactly 1.0 (LongMemEval `oracle`, whose haystack ships only evidence), only the recall charts are planned and the document says why — missing precision/F1 charts there are not a bug. `amb plot all --metric` overrides the auto-detection.

Documents are a `Block` tree (`Heading`/`Paragraph`/`Table`/`Figure`/`Rule`) serialized by a `Renderer`. Markdown is the only format registered; adding plain text means implementing `Renderer` and registering it in `RENDERERS`, not touching a single section. Adding a whole new document means a builder plus an entry in `REPORTS` — both CLI commands pick it up. One import rule inside the package: `sections.py` imports from `report.py` at module level, so `report.py` imports its Section classes lazily inside the builder functions — adding a top-level import of `sections` to `report.py` is a circular import.

## Charts

`amb plot scatter --x <metric> --y <metric>` draws the trade-off scatter: one dot per memory system, newest run each, any numeric metric on either axis (dotted paths reach into summary sections, e.g. `search_latency.p50_s`).

Two rules are load-bearing, not cosmetic:

- **Every dot is directly labelled.** Each system's dot wears the hue it keeps on every other chart (sorted-name assignment from the 8-hue categorical palette), but the hue is reinforcement only — identity rests on the label, and label placement dodges collisions. Bars, whose identity sits on the axis, use the single series hue. The k-sweep lines hard-cap at 8 systems: `lines()` raises rather than invent a 9th hue — filter or facet.
- **Points from different datasets or k values are never mixed** on one pair of axes; `amb plot scatter` refuses and asks you to pick (`--dataset`, `--k`). Mode is guarded elsewhere: `amb plot k` refuses mixed modes, and `amb plot all` / `amb report` refuse via the group guard — scatter itself keeps the newest run per system, whatever its mode.

Before changing chart code, load the `dataviz` skill and follow its procedure (form → colour → *run the validator* → marks → accessibility → look at the render). Never eyeball a palette, and always open the image you produced — the validator checks colour, not layout. The first version of this chart looked fine in code and printed six overlapping labels on top of each other.

## System versions

Every run records the version of the system it measured (`system_version` in the summary): the SDK distribution's installed version for in-process systems (`mem0ai`, `graphiti-core`, amb itself for `naive`), and the **server** version for letta and fraise — the server is the system, the client is plumbing; both fall back to the client SDK's dist when the server is unreachable at capture time. `system_version` is part of the comparison identity, so a version upgrade is a new benchmark subject and can never silently displace old results in `amb report --latest`.

Dependencies pin the measured line (`mem0ai>=2.0,<3`, `graphiti-core>=0.29,<0.30`, `letta-client>=1.12,<2`, `fraise-sdk>=0.1.0a1,<0.2`) and the compose images pin exact tags (`letta/letta:0.16.8`, `qdrant/qdrant:v1.19.0`, `neo4j:5.26`, `ghcr.io/ronsenbergvi/fraise:0.1.0-beta.7`). Loose ranges already bit twice — mem0 2.0's breaking API changes produced a 400 and a silently-ignored parameter. Bump pins deliberately, in their own commit, and expect a new results row. Two couplings to know when bumping: letta's token spend is *computed*, not observed — tiktoken arithmetic calibrated against a counting proxy for letta 0.16.8 under two assumptions (one embeddings call per insert/search, no server-side chunking); re-verify both on an image bump or the published cost numbers go silently wrong. And mem0's spaCy model is pinned as a direct wheel URL in `[tool.uv.sources]` (`en_core_web_sm`), which dependency resolution will not bump for you when the `[nlp]` extra moves.

## Logging and diagnosis

Logging is loguru, configured in `logs.py` and always to **stderr**, so the summary on stdout stays pipeable. `amb --log-level debug run ...` traces the turn budget, how many questions survived it, and per question: hit count, how many session/turn ids were retrieved, and the evidence ids they are scored against (the retrieved ids themselves land in the saved `results.jsonl`, not the log).

Retrieval is scored at **every level the dataset's labels and the system's hits both support**, as two separate metric families: session-level `retrieval_precision/recall/f1` — the comparable headline, since every system can attest which session a memory came from — and turn-level `turn_precision/recall/f1`, a stricter bonus for systems that store verbatim turns (naive, letta). The families never share a name, so a session-level number can never be compared with a turn-level one; comparison tables use only the session-level columns. Rows carry the raw ids per level (`retrieved_session_ids`/`evidence_session_ids`, `retrieved_turn_ids`/`evidence_turn_ids`). Hits with no provenance at any labelled level are excluded from retrieval metrics with a warning, instead of scoring a false 0.0 — that silent zero is the single easiest way to publish a wrong number from this repo. If you add a `Memory`, populate `MemoryHit.turn_ids` and/or `session_ids` (letta maps passages to turns; graphiti maps edge episodes to sessions; mem0 attaches session metadata).

## Conventions

- **Data contracts are pydantic models** (`contracts`), validated at the `Run` boundary.
- **The CLI is click**, one command per module under `cli/commands/`, attached from the `COMMANDS` tuple — a new command is a new module plus one entry there, nothing in `main.py`. `--param KEY=VALUE` coerces types (`true`/`false`, numbers) — argparse-style raw strings would make `reasoning=false` truthy.
- **Docker**: databases use named volumes so ingestion survives `down`; wipe with `down -v`. Exception: fraise runs its server without a volume on purpose — the alpha SDK has no delete verb, so teardown is a no-op and a fresh `up` starting empty *is* the wipe; don't add one. Don't set path env vars in a Dockerfile — `WORKDIR /amb` already makes the default relative paths land on the mounted volumes.

## Working practice

- **Measure, don't estimate.** Runtime, dataset size, and cost claims in this repo come from real runs. A progress bar that isn't advancing in `docker logs` is not evidence a run is stuck — check the database or the saved report before concluding anything.
- **Verify against the installed version, not memory.** SDKs here move fast and break APIs (mem0 2.0 moved `user_id` into `filters` and renamed `limit` to `top_k`; the second one failed *silently*). Read the installed source or the wheel before asserting how a library behaves.
- **Never print resolved secrets.** `docker compose config` interpolates `.env` and will dump API keys into the terminal. Mask, or query specific non-secret keys.
- **A truncated or scoped run must be marked as such.** `--turns` records `max_turns` in the summary and forms part of the comparison identity, so a smoke run can never displace a real run in `amb report --latest`. Keep that property when adding new scoping options.
- **Scoping is two separate axes, deliberately.** `--turns` is corpus size (ingestion, the expensive phase); `--questions` is workload (asked after ingestion). Do not merge them — you need to be able to ingest a small corpus and still ask every question of it. The budget is counted in *turns*, the unit both datasets share; sessions are not comparable across them.
- **Never score a question whose evidence was not ingested.** `select_questions` drops them and records `questions_dropped`; scoring them would report a retrieval failure the truncation caused. Questions without evidence labels keep running, since there is no way to tell whether their evidence survived.
- **State what is measured versus extrapolated** in any results or docs.
