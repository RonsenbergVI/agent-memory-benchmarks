# everos

Integration for [EverMind-AI/EverOS](https://github.com/EverMind-AI/EverOS). EverOS runs in-process — memories are Markdown files under `EVEROS_ROOT` with SQLite and LanceDB indexes beside them — so there is no server for compose to stand up.

## gpt-5-mini needs one accommodation

`everalgo` asks for `temperature=0.0` on every call, and EverOS builds its `LLMConfig` from only `model` / `api_key` / `base_url` — so there is no configuration path to change it, and OpenAI's reasoning models answer every call with `'temperature' does not support 0.0`.

`_pin_temperature` lifts the configured temperature to 1 for those models. It is the same accommodation mem0 carries in this repo for the same models and the same 400, and the narrowest one available: everalgo reads `LLMConfig.temperature`, a real field on its own public config, so no everalgo internal is patched and a non-reasoning model is left exactly as EverOS shipped it. It fails loudly if everalgo ever moves that field, rather than silently reverting to 400s. `--param reasoning=false` opts out.

With it, EverOS ingests with `gpt-5-mini` and embeds with `text-embedding-3-small` at 1536 dimensions — model-matched to the rest of the comparison.

Spend **is** made by this system, but see the note on token counting below.

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
| `model` | `gpt-5-mini` | Extraction model, matching the rest of the comparison. |
| `reasoning` | auto | Whether the model rejects `temperature=0.0`. Auto-detected from the model name; `false` restores everalgo's own temperature. |
| `embedding_model` | `text-embedding-3-small` | Embedder, at `embedding_dimensions` (1536). |
| `search_method` | `hybrid` | `hybrid` is vector + keyword. `agentic` adds an LLM planning pass — query understanding rather than retrieval, so not comparable with the other systems. |
| (env) `EVEROS_ROOT` | `.everos` | Where the Markdown and indexes live. |

## Notes

- **Isolation is a project** (`conv-<id>`), which is both the search filter and a directory segment under the memory root — so recall cannot cross conversations, and teardown is a directory removal scoped to one id.
- **One EverOS user per conversation.** EverOS keys user memory by a message's `sender_id` and requires a search to name exactly one of `user_id`/`agent_id`. A LoCoMo conversation has two speakers, but the unit of memory here is the conversation, so both speak as one user; the speaker survives in `sender_name` and in the text of every message.
- **Provenance is session-level.** An episode is an extraction over a session's messages, not a verbatim turn, so hits carry `session_ids` and never claim `turn_ids`.
- **`everalgo-boundary` is pinned.** EverOS pins `everalgo-agent-memory==0.4.0`, which declares `everalgo-boundary<2.0.0,>=0.2.0` — loose enough to admit 0.3.0, where `DetectionResult` gained a required `should_wait` that agent-memory 0.4.0's call site does not pass. It fails at the first `memorize()`, not at install.
