# honcho

Integration for [plastic-labs/honcho](https://github.com/plastic-labs/honcho), via its official `honcho-ai` Python SDK, against a self-hosted server.

The server is AGPL-3.0; the SDK is Apache-2.0. Only the self-hosted edition is measured — the managed API at api.honcho.dev is never contacted, and no Honcho platform key exists anywhere in this stack.

## The server is built from source

Honcho publishes no container image. The compose file builds the project's own Dockerfile straight from the `v3.2.2` tag, and runs it twice — the API and the deriver, Honcho's background reasoning worker — over Postgres + pgvector, as Honcho's own compose does. `system_version` records the server's version from its OpenAPI document, so a bump lands as a new row.

## What is measured

Each conversation is a workspace, each speaker a peer, each dataset session a Honcho session, and each turn a message from its speaker, stamped with the session's in-world date.

**Recall is Honcho's message search** — the workspace `search` endpoint, hybrid semantic + full-text over the stored messages — so every hit is a verbatim turn, searched the way the other systems' stores are. The question travels verbatim.

**Ingest time includes Honcho's reasoning.** Honcho reasons over every message in the background (the deriver writes conclusions about each speaker), and `ingest_session` waits for the session's queue to drain before it returns, so that work is charged to ingestion and never still in flight when the first question is asked. The deriver runs with `DERIVER_FLUSH_ENABLED=true` — as Honcho's own bench harness does — or a short session would sit in its batch for up to 30 minutes waiting for a token threshold.

The deriver runs gpt-5-mini at `THINKING_EFFORT=minimal`, Honcho's template value. Left unset, gpt-5-mini reasons at "medium", spends the deriver's whole 2,500-token output budget thinking, and comes back empty — the deriver only logs it, so a run would look clean while storing nothing.

## Cost is not measured — the column says so

Every LLM and embedding call happens inside the server, and Honcho reports no usage for any of them — no field on a response, no endpoint. The system declares `usage_coverage = "none"`, so its cost renders as unmeasured rather than as a free zero. Counting it would take a proxy in front of the server's OpenAI traffic.

## Off by default, and why

- **Summaries** are disabled per workspace, as Honcho's own bench does. They feed the dialectic, which recall never calls.
- **Dreams** — Honcho's consolidation pass, which derives deductive and inductive conclusions — are disabled per workspace. They fire on idle timers, which would land them at arbitrary points mid-run; Honcho's bench instead schedules them explicitly once ingestion ends, and the harness has no end-of-ingestion hook to do that from.

## Running it

```bash
docker compose -p honcho-smoke -f benchmarks/honcho/docker-compose.yaml \
  run --build --rm benchmark run --system honcho \
  --dataset locomo --limit 1 --turns 40 --questions 5
docker compose -p honcho-smoke -f benchmarks/honcho/docker-compose.yaml down -v
```

The server reads its key as `LLM_OPENAI_API_KEY`, which `env_file` can inject but cannot rename, so both server entrypoints map `OPENAI_API_KEY` across at exec time. That works whether the key comes from `../../.env` locally or from the environment CI exports.

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `queue_timeout` | `900` | Seconds one session's reasoning may take before the run fails rather than querying a half-reasoned store. |
| `base_url` | `HONCHO_BASE_URL`, else `http://localhost:8000` | Which server to measure. |
| `timeout` | `60` | Client timeout in seconds. |

## Notes

- **Provenance is turn-level.** Each message carries its turn id and dataset session id in its metadata, so a hit names both back — scored at session level like every system, and at turn level as well.
- **Isolation is a workspace** (`amb-<conversation>`), so recall cannot cross conversations by construction. Teardown deletes the workspace's sessions, then the workspace — Honcho refuses to delete a workspace with live sessions.
- **The deriver runs three workers**, one per conversation CI runs in parallel, so one conversation's ingest time does not include another's queue. Polling backoff is off: the adapter idles the queue after every session, and backoff would charge up to 30 s of poll sleep to each one.
