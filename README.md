# Agent Memory Benchmark

An open benchmark and reference for evaluating AI agent memory systems.

## What this is?

A neutral, reproducible benchmark for systematically evaluating AI agent memory systems. The repo provides:

- A curated registry of agent memory projects with links, descriptions, and metadata
- Standardized evaluation tasks across long-term recall, temporal reasoning, multi-session continuity, and contradiction handling
- A test harness that runs the same tasks against different memory systems
- Published results with reproducibility instructions
- An open methodology — contributions welcome for new projects, new tasks, and new metrics

This is not a marketing tool for any one system.

## Why this exists?

The AI agent memory space has grown to dozens of distinct projects in the last two years, each making different design choices: vector-based, graph-based, hierarchical, hybrid, summary-based. The trade-offs between them are real and consequential — but there's no neutral place where they're made visible.

Today, evaluation in this space typically takes one of three shapes:

- Vendor-led: Each project's docs benchmark it against carefully-selected baselines designed to make it look favorable.
- Paper-led: Academic benchmarks use specific datasets that don't always translate to production agent behavior.
- Ad-hoc: Users informally compare two or three systems on their own use cases, with no shared methodology.

None of these give a builder a reliable answer to "which agent memory system fits my use case?" — let alone "by how much." This benchmark exists to fix that gap. The goal isn't to pick a winner. It's to make the relevant trade-offs (accuracy vs. latency, recall vs. precision, setup complexity vs. memory footprint) measurable and transparent so practitioners can make informed choices.

## Inclusion criteria

Projects must be permissively licensed (MIT, Apache 2.0, BSD) or use a credible open-source-adjacent license. Hosted SaaS without an OSS counterpart is out of scope. Open-core projects are included via their OSS edition only — commercial/cloud features aren't benchmarked.

## Status

Pre-alpha. Methodology is being defined and the test harness is in development. First results land here on the first tagged release and regenerate on every one after (see [RESULTS.md](RESULTS.md) for the full detail). Expected timeline:

- Methodology v0.1 finalized: TBD
- Test harness v0.1 (first 3 systems running end-to-end): TBD
- First public results report: TBD

## Results

<!-- amb:summary -->
Results from commit `38a4bac`, run 2026-08-20.

Newest run per system at k=10 — full detail in [RESULTS.md](RESULTS.md).

### locomo

| system | version | precision | recall | F1 | memory tokens | p50 search (s) |
| --- | --- | --- | --- | --- | --- | --- |
| graphiti | 0.29.3 | 0.276 | 0.971 | 0.420 | 908,236 | 0.2662 |

![Retrieval F1 vs k](plots/locomo/k_f1.png)

![Retrieval recall vs k](plots/locomo/k_recall.png)

![Retrieval precision vs k](plots/locomo/k_precision.png)

### longmemeval (oracle)

| system | version | precision | recall | F1 | memory tokens | p50 search (s) |
| --- | --- | --- | --- | --- | --- | --- |
| graphiti | 0.29.3 | 1.000 | 0.897 | 0.933 | 1,276,025 | 0.3192 |

![Retrieval recall vs k](plots/longmemeval/oracle/k_recall.png)
<!-- /amb:summary -->

## Projects in scope

The following systems are tracked. Inclusion criteria: agent memory is a primary concern, not an afterthought; project is publicly accessible (OSS or has a public API); reasonable maintenance signal in the last 6 months.

| Project | Repository | Language | License | Memory model | Status |
|---------|------------|----------|---------|--------------|--------|
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | Python | Apache-2.0 | Vector + extraction | 🟢 |
| Letta | [letta-ai/letta](https://github.com/letta-ai/letta) | Python | Apache-2.0 | Hierarchical (OS-style) | 🟢 |
| Graphiti | [getzep/graphiti](https://github.com/getzep/graphiti) | Python | Apache-2.0 | Temporal knowledge graph | 🟢 |
| Cognee | [topoteretes/cognee](https://github.com/topoteretes/cognee) | Python | Apache-2.0 | Graph + vector hybrid | ❓ |
| Memary | [kingjulio8238/memary](https://github.com/kingjulio8238/memary) | Python | MIT | Knowledge graph | ❓ |

## Methodology

The full methodology — design principles, evaluation axes, harness design, datasets, model selection, run scoping, and cost — lives in [METHODOLOGY.md](METHODOLOGY.md).

## Quickstart

```bash
uv sync
uv run amb datasets list            # locomo | longmemeval (+ variants)
uv run amb datasets pull locomo     # download into .data/
uv run amb run --system naive --dataset locomo            # direct mode, no LLM needed
uv run amb run --system naive --dataset locomo \
  --model openai:gpt-5-mini --judge                       # + answer generation & grading
uv run amb report                   # markdown table across all runs
```

Every benchmark is the same `amb` command, with or without Docker. Locally, install the integration's package and point it at a database you run yourself (adapters read connection settings from env vars with localhost defaults):

```bash
uv run --package graphiti amb run --system graphiti --dataset locomo
```

Or let the integration's Docker environment provide the database — the container's entrypoint is `uv run amb`, so everything after `benchmark` is the identical CLI:

```bash
docker compose -f benchmarks/graphiti/docker-compose.yaml run --build --rm benchmark \
  run --system graphiti --dataset locomo
```

## Examples

```bash
# smoke run: naive baseline, 2 LoCoMo conversations, retrieval scoring only
# (BM25 only, no LLM calls — finishes in seconds; use this to check plumbing)
uv run amb run --system naive --dataset locomo --limit 2 --questions 30

# agentic run: the model drives ingestion and search through the system's
# own tools; --model is mandatory — an agentic run never falls back
uv run amb run --system naive --dataset locomo --mode agentic \
  --model openai:gpt-5-mini

# mem0 in docker, smoke test: 2 sessions of one conversation, 3 questions
# (needs OPENAI_API_KEY for mem0's extraction; compose reads the repo-root .env)
docker compose -f benchmarks/mem0/docker-compose.yaml run --build --rm benchmark \
  run --system mem0 --dataset locomo --limit 1 --turns 40 --questions 3

# trade-off plot: one dot per system, any metric on each axis
uv run amb plot scatter --x search_latency.p50_s --y retrieval_recall --dataset locomo

# draw every chart the generated reports link, into plots/<dataset>[/<variant>]
uv run amb plot all

# regenerate RESULTS.md (written whole) and README.md's results section
# (spliced); the release CI runs both of these on every tag
uv run amb report --latest --output RESULTS.md --summary README.md --tag v0.1.0
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the contribution process. Roughly:

- Bug reports and improvements to existing benchmarks: open an issue.
- New adapters for systems already in scope: open a PR adding `benchmarks/<system>/`.
- New systems to add: open an issue first to discuss inclusion criteria.
- Methodology critique or improvements: open a discussion in [GitHub Issues](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues). This benchmark only works if the methodology is trusted. We take methodology critique seriously and document responses to substantive criticism publicly.
