# Contributing to agent memory benchmarks

Development commands live in the Makefile (`make help`); run `make check` before committing and `make smoke` after touching the run loop. The harness itself is covered by `uv run pytest` (fixture-based, no network).

## Adding a memory framework

A framework is a workspace package under `benchmarks/<name>/` — no core changes. Use `benchmarks/fraise/` as the reference; the naive baseline in `src/amb/memory.py` is the minimal in-core example of the same contracts.

1. **`benchmarks/<name>/pyproject.toml`** — a package named `<name>` depending on `amb` (workspace source) and the system's SDK. Every integration ships its module as `src` (SDKs own the pretty import names), so the build packages `["src"]` and the entry point is:

   ```toml
   [project.entry-points."amb.systems"]
   <name> = "src.benchmark:<Name>Benchmark"

   [tool.uv.sources]
   amb = { workspace = true }
   ```

   Keep the `# x-release-please-version` comment on the version line — release-please bumps it in lockstep with the root package. The root workspace picks up `benchmarks/*` automatically; run `uv lock` after adding the package.

2. **`benchmarks/<name>/src/memory.py`** — subclass `amb.base.Memory`: set the `name`, `description`, and `sdk_dist` class attributes and implement `ingest_session` and `search` (plus `setup`/`teardown` for connect/cleanup and `stats()` for footprint numbers). Two contracts matter more than they look:

   - **Provenance.** Every `MemoryHit` must carry `session_ids` (the comparable headline every system can attest) and, when the system stores verbatim turns, `turn_ids`. Retrieval is scored against these; hits without provenance drop the question from retrieval metrics rather than scoring a false 0.0.
   - **Spend.** LLM/embedding calls made in-process through the openai SDK are tracked automatically. Spend that happens inside the system's own server is invisible — count it yourself and report it through `usage_counters()`, with `TiktokenUsageTracker` in the benchmark's `callback_classes` to book the deltas.

   Models are identity: expose the ingestion/embedding models as `model` / `embedding_model` attributes (the default `models()` reads them) so runs with different models land as different rows.

3. **`benchmarks/<name>/src/toolset.py`** — the agentic-mode tool surface: subclass `SearchToolset` and `IngestToolset` from `amb.agent.toolset`, add the system's native tools with `add_function`, and report every search through `self.record(hits, seconds)` and every write through `self.record_write(seconds)`. Validate agent citations against `self.turn_ids()`. The base classes bind the store as `self.memory` — don't shadow that name with a property; use a concrete-typed property named after the system (see `NaiveIngestToolset.naive`). Skip this file entirely for a system with no agentic surface.

4. **`benchmarks/<name>/src/benchmark.py`** — subclass `amb.base.Benchmark`: set `name` and `system_class`, plus `search_toolset_class` / `ingest_toolset_class` (both present = agentic mode supported), `default_params`, and `callback_classes`. Override the factory hooks (`create_system`, `create_search_toolset`, `create_ingest_toolset`, `before_sample`, `after_sample`) only when the system needs objects built its own way.

5. **`benchmarks/<name>/Dockerfile` + `docker-compose.yaml`** — copy an existing pair. The Dockerfile is `uv sync --package <name>` from the repo root; compose adds whatever server/database the system needs, volume-mounts `../../.data:/amb/.data` and `../../report:/amb/report`, and passes keys via `env_file: ../../.env` — never bake keys into the image.

6. **`.github/workflows/<name>.yml`** — copy `fraise.yml` and swap the framework name and compose path (drop the ghcr login if the image isn't on ghcr). One job per dataset; matrices hold plain values only.

7. **Repo plumbing** — three configs track packages by name and must learn the new one:

   - `.github/release-please-config.json` — the version annotation from step 1 only works if the file is registered: add `benchmarks/<name>/pyproject.toml` to `extra-files`, plus a typed entry that bumps the package's pin in the lockfile — without it the release PR leaves `uv.lock` stale and fails CI's `uv sync --locked`:

     ```json
     {
       "type": "toml",
       "path": "uv.lock",
       "jsonpath": "$.package[?(@.name=='<name>')].version"
     }
     ```

   - `.github/dependabot.yaml` — a `package-ecosystem: uv` block for `directory: /benchmarks/<name>`, copied from an existing one (daily schedule, `deps:` commit prefix, labels `area:benchmarks` + `memory:<name>`).
   - `.github/labeler.yaml` — a `memory:<name>` entry globbing `benchmarks/<name>/**`, so PRs touching the integration get labeled. Create the `memory:<name>` label in the repo as well (`gh label create`): dependabot silently ignores labels that don't exist.

Check it landed: `uv run amb systems` lists the entry point, and `docker compose -f benchmarks/<name>/docker-compose.yaml run --build --rm benchmark run --system <name> --dataset locomo --limit 1 --turns 40 --questions 5` runs the whole pipeline on one conversation.

External packages work too: anything installed in the environment that registers the `amb.systems` entry point shows up in `amb systems`.

## Adding a dataset

A dataset is a loader in core — one module, one enum value, one registry entry. Use `src/amb/datasets/locomo.py` as the reference.

1. **`src/amb/constants.py`** — add the dataset to the `Dataset` enum.
2. **`src/amb/datasets/<name>.py`** — subclass `amb.base.DatasetLoader`: set `name` (the enum value) and, when the dataset has size/split options, `variants` and `default_variant`. Implement:
   - `pull(variant)` — download the raw files (HuggingFace or URL) into `self.cache_dir`, idempotently, and return the local path.
   - `load(variant, limit)` — normalize the raw payload into `Sample`s: a `Conversation` of `Session`s of `Turn`s, plus `QAPair`s carrying the gold answer and the evidence labels (`evidence_session_ids`, and `evidence_turn_ids` where the dataset has them) that retrieval is scored against. `limit` caps the number of samples for smoke runs. Set `QAPair.category` — categories are free-form strings and every summary gets a per-category breakdown automatically.
3. **`src/amb/datasets/__init__.py`** — add the loader to the `LOADERS` dict. That's the whole registration: `amb datasets list` shows it, `amb datasets pull <name>` caches it, and every system can run it via `amb run --dataset <name>`.
4. **`tests/datasets/<name>_test.py`** — pin the schema mapping with a small raw-payload fixture (no network), like the existing dataset tests.
5. **Workflows** — each framework workflow gets a new job for the dataset (one job per dataset, plain-value matrix), mirroring the existing jobs in `fraise.yml`.

Variants are identity: a dataset run with `--variant` lands under `runs/<dataset>/<variant>/...` and is grouped as its own experiment in every table and chart — use variants for differently-shaped haystacks, not for scope (that's what `--limit`/`--turns`/`--questions` are for).

## Extending the evaluation

- **New metric.** Add a `Metric` subclass to `src/amb/metrics.py` and put it in `default_metrics()`; dots in a metric name nest its result into a summary section (`ingest.total_s` → `summary["ingest"]["total_s"]`). Reporting is data-driven — any top-level float in a run summary becomes a column in the comparison table, and any summary key is plottable via `amb plot scatter --x/--y`. Record new per-question fields in `Runner._run_question` (or per run in `RunReport.summary()`).
- **New report or chart set.** Reports are declared in `src/amb/reporting/report.py` (`REPORTS`) from sections in `sections.py` — a section declares its charts and renders its blocks from the same list, so documents and `amb plot all` cannot drift apart.
