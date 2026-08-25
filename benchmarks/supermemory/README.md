# supermemory

Integration for [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory), via its official Python SDK.

## Cost is not measured for this system

Every other integration's spend is observable: it either runs in-process through the openai SDK (tracked automatically) or inside a server this repo stands up (counted and reported through `usage_counters()`). Supermemory's extraction runs inside Supermemory's own infrastructure, against a provider this harness never touches — the SDK-patching tracker sees nothing because nothing goes through our client, and the counting proxy sees nothing because the traffic never traverses it.

A supermemory run therefore records `memory_tokens_total: 0`. **That zero means unmeasured, not free.** It is not comparable with another system's measured zero (mempalace, which genuinely makes no API call) or with a measured spend. Read this system's rows on the cost axes as "no data".

Pointing the adapter at a self-hosted Supermemory local server whose LLM provider is the counting proxy would make the spend measurable; nothing here does that today.

## Running it

`SUPERMEMORY_API_KEY` is required by both deployments.

- **Hosted API** (default) — no server to stand up; `SUPERMEMORY_API_KEY` is the account's key.
- **Supermemory local** — install per the [self-hosting docs](https://supermemory.ai/docs/self-hosting/overview), start `supermemory-server`, and set `SUPERMEMORY_BASE_URL=http://localhost:6767` plus the key it prints on first boot. The two speak the same API, which is why one adapter covers both; there is no compose service for it because the project ships no container image and its first boot is an interactive wizard.

```bash
docker compose -f benchmarks/supermemory/docker-compose.yaml \
  run --build --rm benchmark run --system supermemory \
  --dataset locomo --limit 1 --turns 40 --questions 5
```

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `search_mode` | `memories` | `memories` retrieves extracted memories (the analogue of every other system's search); `documents` returns whole source documents; `hybrid` mixes the two. |
| `rerank` | `true` | Supermemory's own reranking pass. `false` measures the raw retrieval underneath it. |
| `index_timeout` | `600` | Seconds one document may take to become searchable before the run fails. |
| `base_url` | `SUPERMEMORY_BASE_URL`, else the hosted API | Which server to measure. |

## Notes

- **Writes are asynchronous.** `documents.add` returns as soon as the document is queued; its memories are not searchable until it reaches `done`. The adapter follows every write to a terminal status before the next one, so the measured ingestion time is the real one and the first question never runs against a store that has not caught up.
- **Provenance is session-level.** A returned memory is an extraction over a document, not a verbatim turn, so hits carry `session_ids` and never claim `turn_ids` — the same level as mem0 and graphiti.
- **Isolation is a container tag** (`conv-<id>`), Supermemory's own scoping primitive and a hard filter on both write and search.
