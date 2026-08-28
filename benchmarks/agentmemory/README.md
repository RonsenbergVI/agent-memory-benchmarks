# agentmemory

Integration for [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory) — the Node memory server, not the unrelated PyPI package of the same name.

## Isolation is not exact under `--workers N`

agentmemory accepts a `project` on every call, but `mem::smart-search` never passes it to the searcher — it scopes lesson recall only. The dimension search *does* filter on is `agentId`, which an observation inherits from its session row, which takes it from `session/start`. So this adapter sends the conversation id as the agent id at session start and as the filter on every search.

That filter is applied **after retrieval, not inside the index** — upstream's own comment says the BM25 and vector indexes do not carry the agent id, so the server over-fetches 3× the requested limit and trims. With several conversations resident at once, one conversation's true top-k can be crowded out of that window by another's hits before the filter runs.

The adapter asks for `k × 10` (capped at the API's 100) and trims to k locally to widen the window, but it cannot close it. **`--workers 1` is the only setting where isolation is exact**, because the store then holds one conversation at a time — teardown forgets each conversation's sessions before the next begins. The compose command sets it, and the workflow inherits it. Raising it trades a measurement guarantee for wall clock; if you do, say so on the results row.

## What it measures

The server runs with LLM compression on (`AGENTMEMORY_AUTO_COMPRESS=true`) and OpenAI embeddings, so its configuration matches the rest of the comparison rather than sitting in agentmemory's keyless BM25-only default.

The extraction model is `gpt-5-mini`, matching the rest of the comparison — but that needs a build-time patch. agentmemory sends `max_tokens` in its OpenAI request and never `max_completion_tokens`, the only form OpenAI's reasoning models accept, and exposes no setting for it. On gpt-5-mini every compression call returns 400, and the failure is destructive rather than degraded: the observation is dropped without ever being indexed, so the system retrieves nothing at all.

`patch-openai-params.js` renames the parameter in the OpenAI provider at image build time, located by its Azure-aware `buildChatUrl` call. The Anthropic, OpenRouter and MiniMax providers share the same bundle and keep `max_tokens`, which is correct for them. The script asserts before and after, so a version bump that moves the call fails the build rather than silently restoring the 400s.

`CONSOLIDATION_ENABLED=false` keeps consolidation off — that is a separate periodic pipeline, not extraction.

Ingestion uses agentmemory's own hook path: one agentmemory session per dataset session, and one `prompt_submit` observation per turn — the shape the project's Claude Code hooks produce, and the only hook whose payload is an utterance rather than a tool call or a lifecycle event.

## Running it

```bash
docker compose -p agentmemory-smoke -f benchmarks/agentmemory/docker-compose.yaml \
  run --build --rm benchmark run --system agentmemory \
  --dataset locomo --limit 1 --turns 40 --questions 5
docker compose -p agentmemory-smoke -f benchmarks/agentmemory/docker-compose.yaml down -v
```

There is no published agentmemory image, so `server.Dockerfile` builds one from the project's own deploy template (`deploy/fly/` upstream): `node:22-slim`, the `iii` binary copied out of the official `iiidev/iii` image, and the npm package. `AGENTMEMORY_VERSION` / `III_VERSION` / `III_SDK_VERSION` are build args.

**The iii pin matters.** agentmemory's caret range resolves `iii-sdk` to 0.11.6, which needs a sandboxed-worker model agentmemory has not been refactored for. The mismatch does not fail at startup — it surfaces as EPIPE reconnect loops and *empty search after save*, which in a benchmark reads as a system that retrieves nothing. The `overrides` block in the server image's `package.json` is what holds it at 0.11.2; don't drop it when bumping.

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `base_url` | `AGENTMEMORY_BASE_URL`, else `http://localhost:3111` | Which server to measure. |
| `secret` | `AGENTMEMORY_SECRET` | Bearer token, when the server requires one. |
| `include_lessons` | `false` | Lessons are an LLM-derived object type a keyless server never produces; asking costs a lookup per search and returns nothing. |
| `timeout` | `120` | HTTP timeout in seconds. |

## Notes

- **Search is two round trips by construction.** The compact search ranks and returns `obsId`/`score`/`sessionId` with no content; a second call expands those ids into their observations. Both are charged as one search, which is what the system costs to answer one question. `expandIds` is capped at 20 server-side — every k this harness sweeps is inside it.
- **Provenance is exact at both levels.** One observation is one turn, so `obsId` maps back to the turn that produced it, and the hit's `sessionId` carries the dataset session. Session ids are scoped `<conversation>:<session>` on the way in, because agentmemory's session ids are global to the server while the datasets' are only unique inside a conversation.
- **Deduplicated writes claim no provenance.** A repeated utterance is answered with `{"deduplicated": true}` and no id; the turn is genuinely not separately stored, so the adapter does not record a turn mapping the store could never return.
