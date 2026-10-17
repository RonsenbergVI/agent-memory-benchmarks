# Agent Memory Benchmark

<div align="center">

[![CI](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/RonsenbergVI/agent-memory-benchmarks?filter=v*)](https://github.com/RonsenbergVI/agent-memory-benchmarks/releases)
[![License](https://img.shields.io/github/license/RonsenbergVI/agent-memory-benchmarks)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)

[![Benchmark cognee locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/cognee-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/cognee-locomo.yml)
[![Benchmark cognee longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/cognee-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/cognee-longmemeval.yml)
[![Benchmark everos locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/everos-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/everos-locomo.yml)
[![Benchmark everos longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/everos-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/everos-longmemeval.yml)
[![Benchmark fraise locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/fraise-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/fraise-locomo.yml)
[![Benchmark fraise longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/fraise-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/fraise-longmemeval.yml)
[![Benchmark graphiti locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/graphiti-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/graphiti-locomo.yml)
[![Benchmark graphiti longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/graphiti-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/graphiti-longmemeval.yml)
[![Benchmark hindsight locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/hindsight-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/hindsight-locomo.yml)
[![Benchmark hindsight longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/hindsight-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/hindsight-longmemeval.yml)
[![Benchmark langmem locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/langmem-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/langmem-locomo.yml)
[![Benchmark langmem longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/langmem-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/langmem-longmemeval.yml)
[![Benchmark mem0 locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/mem0-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/mem0-locomo.yml)
[![Benchmark mem0 longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/mem0-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/mem0-longmemeval.yml)
[![Benchmark memos locomo](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/memos-locomo.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/memos-locomo.yml)
[![Benchmark memos longmemeval](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/memos-longmemeval.yml/badge.svg)](https://github.com/RonsenbergVI/agent-memory-benchmarks/actions/workflows/memos-longmemeval.yml)

**[Results](#results) · [Quickstart](#quickstart) · [Why this exists](#why-this-exists) · [Methodology](#methodology) · [Projects in scope](#projects-in-scope) · [Contributing](#contributing)**

</div>

An open benchmark and reference for evaluating AI agent memory systems

## Results

Every system is measured the way an agent uses memory: it ingests each conversation through its own extractor, an agent searches it through the system's own tools and answers, and an LLM judge marks every answer correct or not. The headline is that judge accuracy, with its 95% interval; retrieval recall sits beside it as a diagnostic. The earlier retrieval-only results measured a search engine rather than memory in an agent, and were withdrawn.

Every image below is generated by the release CI into [plots/](plots/) — a guarded directory that pull requests cannot touch — and regenerates on every benchmark run, each figure carrying its own run date. The full k-sweep lives in [RESULTS.md](RESULTS.md), guarded the same way. The charts appear with the first release run.

### locomo

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/summary_k10_dark.png">
  <img alt="Cross-system summary at k=10: judge accuracy with its 95% interval" src="plots/locomo/summary_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/accuracy_k10_dark.png">
  <img alt="Judge accuracy by system at k=10, with 95% Wilson intervals" src="plots/locomo/accuracy_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/categories_k10_dark.png">
  <img alt="Judge accuracy by question category at k=10" src="plots/locomo/categories_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_dark.png">
  <img alt="Judge accuracy vs k" src="plots/locomo/k_accuracy.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_recall_dark.png">
  <img alt="Retrieval recall vs k" src="plots/locomo/k_recall.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_adversarial_dark.png">
  <img alt="Judge accuracy vs k for adversarial questions" src="plots/locomo/k_accuracy_adversarial.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_multi-hop_dark.png">
  <img alt="Judge accuracy vs k for multi-hop questions" src="plots/locomo/k_accuracy_multi-hop.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_open-domain_dark.png">
  <img alt="Judge accuracy vs k for open-domain questions" src="plots/locomo/k_accuracy_open-domain.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_single-hop_dark.png">
  <img alt="Judge accuracy vs k for single-hop questions" src="plots/locomo/k_accuracy_single-hop.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/locomo/k_accuracy_temporal_dark.png">
  <img alt="Judge accuracy vs k for temporal questions" src="plots/locomo/k_accuracy_temporal.png">
</picture>

### longmemeval

LongMemEval's question types at about LoCoMo's ingest time: the `xs` cut asks 151 questions from LongMemEval `s`, each over its own evidence sessions. How the cut is made is in [METHODOLOGY.md](METHODOLOGY.md#longmemeval). These charts appear once the `<system>-longmemeval` workflows run.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/summary_k10_dark.png">
  <img alt="Cross-system summary at k=10: judge accuracy with its 95% interval" src="plots/longmemeval/xs/summary_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/accuracy_k10_dark.png">
  <img alt="Judge accuracy by system at k=10, with 95% Wilson intervals" src="plots/longmemeval/xs/accuracy_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/categories_k10_dark.png">
  <img alt="Judge accuracy by question category at k=10" src="plots/longmemeval/xs/categories_k10.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_dark.png">
  <img alt="Judge accuracy vs k" src="plots/longmemeval/xs/k_accuracy.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_recall_dark.png">
  <img alt="Retrieval recall vs k" src="plots/longmemeval/xs/k_recall.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_knowledge-update_dark.png">
  <img alt="Judge accuracy vs k for knowledge-update questions" src="plots/longmemeval/xs/k_accuracy_knowledge-update.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_multi-session_dark.png">
  <img alt="Judge accuracy vs k for multi-session questions" src="plots/longmemeval/xs/k_accuracy_multi-session.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_single-session-assistant_dark.png">
  <img alt="Judge accuracy vs k for single-session-assistant questions" src="plots/longmemeval/xs/k_accuracy_single-session-assistant.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_single-session-preference_dark.png">
  <img alt="Judge accuracy vs k for single-session-preference questions" src="plots/longmemeval/xs/k_accuracy_single-session-preference.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_single-session-user_dark.png">
  <img alt="Judge accuracy vs k for single-session-user questions" src="plots/longmemeval/xs/k_accuracy_single-session-user.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="plots/longmemeval/xs/k_accuracy_temporal-reasoning_dark.png">
  <img alt="Judge accuracy vs k for temporal-reasoning questions" src="plots/longmemeval/xs/k_accuracy_temporal-reasoning.png">
</picture>

## Quickstart

```bash
uv sync
uv run amb datasets list            # locomo | longmemeval (+ variants)
uv run amb datasets pull locomo     # download into .data/
uv run amb run --system naive --dataset locomo --model test   # offline agent, no API key
uv run amb run --system naive --dataset locomo \
  --model anthropic:claude-haiku-5-5 --judge                  # real agent + judge
uv run amb report                   # markdown table across all runs
```

The `naive` baseline is BM25 only — no API key, no Docker, finishes in minutes. It is the fastest way to see the whole pipeline run end to end.

## What this is

A neutral, reproducible benchmark for systematically evaluating AI agent memory systems. The repo provides:

- A curated registry of agent memory projects with links, descriptions, and metadata
- Standardized evaluation tasks across long-term recall, temporal reasoning, multi-session continuity, and contradiction handling
- A test harness that runs the same tasks against different memory systems
- Published results with reproducibility instructions
- An open methodology — contributions welcome for new projects, new tasks, and new metrics

This is not a marketing tool for any one system.

## Why this exists

The AI agent memory space has grown to dozens of distinct projects in the last two years, each making different design choices: vector-based, graph-based, hierarchical, hybrid, summary-based. The trade-offs between them are real and consequential — but there's no neutral place where they're made visible.

Today, evaluation in this space typically takes one of three shapes:

- Vendor-led: Each project's docs benchmark it against carefully-selected baselines designed to make it look favorable.
- Paper-led: Academic benchmarks use specific datasets that don't always translate to production agent behavior.
- Ad-hoc: Users informally compare two or three systems on their own use cases, with no shared methodology.

None of these give a builder a reliable answer to "which agent memory system fits my use case?" — let alone "by how much." This benchmark exists to fix that gap. The goal isn't to pick a winner. It's to make the relevant trade-offs (accuracy vs. latency, recall vs. precision, setup complexity vs. memory footprint) measurable and transparent so practitioners can make informed choices.

## Inclusion criteria

Projects must be permissively licensed (MIT, Apache 2.0, BSD) or use a credible open-source-adjacent license. Hosted SaaS without an OSS counterpart is out of scope. Open-core projects are included via their OSS edition only — commercial/cloud features aren't benchmarked.

## Datasets

**Running:** LoCoMo · LongMemEval *(coming)* · ConvoMem *(planned)*

## Status

Pre-alpha. Methodology is being defined and the test harness is in development. First results land here on the first tagged release and regenerate on every one after (see [RESULTS.md](RESULTS.md) for the full detail). Expected timeline:

- Methodology v0.1 finalized: TBD
- Test harness v0.1 (first 3 systems running end-to-end): TBD
- First public results report: TBD

## Projects in scope

### Benchmarked

| System | Repository | License | Memory model | Results |
| --- | --- | --- | --- | --- |
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | Apache-2.0 | Vector + extraction | [RESULTS.md](RESULTS.md) |
| Graphiti | [getzep/graphiti](https://github.com/getzep/graphiti) | Apache-2.0 | Temporal knowledge graph | [RESULTS.md](RESULTS.md) |
| Fraise | [RonsenbergVI/fraise](https://github.com/RonsenbergVI/fraise) | MIT | Temporal knowledge graph + vector / text hybrid search | [RESULTS.md](RESULTS.md) |
| EverOS (EverMind) | [EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS) | Apache-2.0 | Local-first, Markdown-native | [RESULTS.md](RESULTS.md) |
| Hindsight | [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) | MIT | Memory units + hybrid search | [RESULTS.md](RESULTS.md) |
| Cognee | [topoteretes/cognee](https://github.com/topoteretes/cognee) | Apache-2.0 | Embedded knowledge graph | [RESULTS.md](RESULTS.md) |
| LangMem | [langchain-ai/langmem](https://github.com/langchain-ai/langmem) | MIT | LLM extraction over a LangGraph store | [RESULTS.md](RESULTS.md) |
| MemOS (MemTensor) | [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | Apache-2.0 | MemCube of extracted memories, held as a graph | [RESULTS.md](RESULTS.md) |

## Methodology

The full methodology — design principles, evaluation axes, harness design, datasets, model selection, run scoping, and cost — lives in [METHODOLOGY.md](METHODOLOGY.md).

## Running locally and in Docker

Every benchmark is the same `amb` command, with or without Docker. Locally, install the integration's package and point it at a database you run yourself (adapters read connection settings from env vars with localhost defaults):

```bash
uv run --package graphiti amb run --system graphiti --dataset locomo --model anthropic:claude-haiku-5-5
```

Or let the integration's Docker environment provide the database — the container's entrypoint is `uv run amb`, so everything after `benchmark` is the identical CLI:

```bash
docker compose -f benchmarks/graphiti/docker-compose.yaml run --build --rm benchmark \
  run --system graphiti --dataset locomo --model anthropic:claude-haiku-5-5
```

## Examples

```bash
# smoke run: naive baseline, 2 LoCoMo conversations, pydantic-ai's offline
# test model as the agent (no LLM calls — use this to check plumbing)
uv run amb run --system naive --dataset locomo --model test --limit 2 --questions 30

# a real run: the system ingests with its own extractor, then amb's question
# agent searches through the system's own tools and the judge grades each
# answer correct or not (needs ANTHROPIC_API_KEY)
uv run amb run --system naive --dataset locomo \
  --model anthropic:claude-haiku-5-5 --judge-model anthropic:claude-haiku-5-5 --judge

# mem0 in docker, smoke test: 2 sessions of one conversation, 3 questions
# (needs OPENAI_API_KEY for embeddings and ANTHROPIC_API_KEY for extraction
# and the agent; compose reads the repo-root .env)
docker compose -f benchmarks/mem0/docker-compose.yaml run --build --rm benchmark \
  run --system mem0 --dataset locomo --llm anthropic:claude-haiku-5-5 \
  --model anthropic:claude-haiku-5-5 --limit 1 --turns 40 --questions 3

# trade-off plot: one dot per system, any metric on each axis
uv run amb plot scatter --x search_latency.p50_s --y retrieval_recall --dataset locomo

# draw every chart the generated reports link, into plots/<dataset>[/<variant>]
# (the release CI runs this twice per tag: light, then --dark for the
# theme-aware images the README embeds)
uv run amb plot all

# regenerate RESULTS.md (written whole); the release CI runs this on every
# tag — README is hand-written and embeds the generated images by path
uv run amb report --latest --output RESULTS.md --tag v0.1.0
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the contribution process. Roughly:

- Bug reports and improvements to existing benchmarks: open an issue.
- New adapters for systems already in scope: open a PR adding `benchmarks/<system>/`.
- New systems to add: open an issue first to discuss inclusion criteria.
- Methodology critique or improvements: open a discussion in [GitHub Issues](https://github.com/RonsenbergVI/agent-memory-benchmarks/issues). This benchmark only works if the methodology is trusted. We take methodology critique seriously and document responses to substantive criticism publicly.

## Support this benchmark

Running this costs real money: every official number comes from a CI run on
tagged code, and the API spend is accounted per-run in each `summary.json`.

If the results are useful to you or your team, you can fund the run pool
through [GitHub Sponsors](https://github.com/sponsors/RonsenbergVI). API
credits count and are disclosed exactly like cash.

**Sponsorship buys nothing about results.** No influence on methodology,
configuration, datasets, run timing, presentation, or which systems are
included — and there is no human step in the pipeline where influence could
apply. The full policy, written before any sponsor existed, is in
[SPONSORS.md](./SPONSORS.md).

## Citing

If you find this work useful, please use the citation:

```bibtex
@software{corneille2026agentmemorybenchmarks,
  author  = {Corneille, Ren{\'e}-Jean},
  title   = {agent-memory-benchmarks: an open benchmark for {AI} agent
             memory systems},
  year    = {2026},
  url     = {https://github.com/RonsenbergVI/agent-memory-benchmarks},
  version = {0.0.1},
  note    = {Results generated by CI from tagged code; all run artifacts public}
}
```
