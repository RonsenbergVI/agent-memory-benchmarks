# everos

Integration for [EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS). EverOS runs in-process — memories are Markdown files under `EVEROS_ROOT` with SQLite and LanceDB indexes beside them — so there is no server for compose to stand up.

## The ingestion model is not gpt-5-mini

Every other integration ingests with `gpt-5-mini`. EverOS cannot: `everalgo` hard-codes `temperature=0.0` and EverOS builds its `LLMConfig` from only `model` / `api_key` / `base_url`, so there is no configuration path to change it. OpenAI's reasoning models reject any temperature but 1 with a 400, which makes gpt-5-mini unusable here as shipped.

The default is therefore `gpt-4.1-mini` — EverOS's own. `--param model=` takes any non-reasoning model, and the run records whichever ran, so a different choice lands as its own row rather than displacing this one. Embeddings are `text-embedding-3-small` at 1536 dimensions, matching the rest of the comparison.

Spend **is** measured: EverOS reaches OpenAI in-process through the openai SDK, so `OpenAIUsageTracker` sees every call.

## Three things the adapter has to do that a server would do for you

EverOS's runtime is normally brought up by its HTTP app's lifespan. Driving the service layer directly skips all of it, and the failures surface one at a time and late:

1. **Runtime startup.** `_start_runtime` runs EverOS's own eight lifespan providers in EverOS's own order against a throwaway app — SQLite schema, LanceDB, LLM and parser clients, cascade, OME engine. Reimplementing the pieces individually produced `no such table: unprocessed_buffer`, then `emit: engine not started`; running the project's own providers means anything it adds later comes along for free.
2. **Config scaffolding.** The OME engine refuses to start without `ome.toml`. `_scaffold_config` copies the same two templates `everos init` does; the models still come from the environment, which outranks the TOML.
3. **Driving the index.** Extraction writes Markdown and records a pending change; the **cascade** is what turns that into the rows search reads, and it normally runs on a filesystem watcher and a schedule. A benchmark ingests and queries immediately, so `_sync_index` drives `sync_once` + `drain_once` after every write. Without it, ingestion succeeds, `extracted_sessions` counts up — and every search returns nothing.

## A write does not necessarily extract

`memorize` accumulates messages and only runs the extraction pipeline when its boundary detector says a topic ended; the return says `accumulated` or `extracted`. Each dataset session is flushed (`is_final=True`) once its turns are in, which is both correct and faithful — a dataset session *is* a conversation boundary.

## Running it

```bash
docker compose -p everos-smoke -f benchmarks/everos/docker-compose.yaml \
  run --build --rm benchmark run --system everos \
  --dataset locomo --limit 1 --turns 40 --questions 5
```

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `model` | `gpt-4.1-mini` | Extraction model. Must be non-reasoning — see above. |
| `embedding_model` | `text-embedding-3-small` | Embedder, at `embedding_dimensions` (1536). |
| `search_method` | `hybrid` | `hybrid` is vector + keyword. `agentic` adds an LLM planning pass — query understanding rather than retrieval, so not comparable with the other systems. |
| `root` | `EVEROS_ROOT`, else `.everos` | Where the Markdown and indexes live. |

## Notes

- **Isolation is a project** (`conv-<id>`), which is both the search filter and a directory segment under the memory root — so recall cannot cross conversations, and teardown is a directory removal scoped to one id.
- **One EverOS user per conversation.** EverOS keys user memory by a message's `sender_id` and requires a search to name exactly one of `user_id`/`agent_id`. A LoCoMo conversation has two speakers, but the unit of memory here is the conversation, so both speak as one user; the speaker survives in `sender_name` and in the text of every message.
- **Provenance is session-level.** An episode is an extraction over a session's messages, not a verbatim turn, so hits carry `session_ids` and never claim `turn_ids`.
- **`everalgo-boundary` is pinned.** EverOS pins `everalgo-agent-memory==0.4.0`, which declares `everalgo-boundary<2.0.0,>=0.2.0` — loose enough to admit 0.3.0, where `DetectionResult` gained a required `should_wait` that agent-memory 0.4.0's call site does not pass. It fails at the first `memorize()`, not at install.
