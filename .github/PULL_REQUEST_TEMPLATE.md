# Pull Request

## What does this PR do?

<!-- One or two sentences. Link an issue if there is one. -->

## Kind of change

- [ ] New memory system integration (`benchmarks/<name>/`)
- [ ] Fix to an existing integration
- [ ] New dataset / dataset loader change
- [ ] New metric / evaluation axis
- [ ] Harness (CLI, reporting, plotting, CI)
- [ ] Docs only

## New memory system checklist (skip if not applicable)

- [ ] `benchmarks/<name>/pyproject.toml` registers `<name>` under the `amb.systems` entry-point group
- [ ] `memory.py` implements `setup`, `ingest_session`, `search`, `teardown`
- [ ] `benchmark.py` subclasses `Benchmark` with `name` and `system_class` set
- [ ] `Dockerfile` + `docker-compose.yaml` added, following an existing integration as the reference
- [ ] `uv run --package <name>-benchmark amb systems` lists it

## Testing

<!-- The exact command(s) you ran to verify this, so a reviewer (or CI) can reproduce -->

```bash

```

- [ ] `uv run pytest` passes
- [ ] Ran the change against real data, not just unit tests (paste the command + a snippet of output above)

## Results / report hygiene

- [ ] This PR does **not** hand-edit `runs/`, `plots/`, or `RESULTS.md`. Those are only ever
      written by `amb report` / `amb plot all`, and only committed by the release CI workflow,
      never by hand (CI enforces this). README.md is hand-written — its results section embeds
      images from `plots/` by stable path and needs no machine edits
- [ ] If this changes what a metric measures, existing numbers in `RESULTS.md` may now be stale —
      noted below, or flagged for a re-run
