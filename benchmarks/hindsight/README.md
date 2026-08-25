# hindsight

Integration for [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight), via its official `hindsight-client` Python SDK.

Note the repo: the project moved to the `vectorize-io` org, and the `hindsight-ai/hindsight` URL in the root README's candidate table is dead. The PyPI name `hindsight` is an unrelated project; the client is `hindsight-client`.

## The server is built from PyPI, not from the project's image

The project's published container images stop at **0.6.2** while the source and the `hindsight-api` distribution are both at **0.9.x**. That gap is not cosmetic: 0.6.2 rejects the `pg_search` text-search backend the project's own compose files recommend, so an image-based stack is forced onto Postgres-native FTS — a materially weaker text half than real BM25.

`server.Dockerfile` therefore pip-installs `hindsight-api` at a pinned version (0.9.2) on `python:3.13-slim`, and the stack runs ParadeDB (pgvector + pg_search) as the project intends. `system_version` records what ran, so a later bump lands as a new row rather than displacing this one.

## Cost is not measured for this system

Hindsight extracts inside its own server, calling the provider that server is configured with (`HINDSIGHT_API_LLM_PROVIDER` / `_MODEL` / `_API_KEY`, set by compose to `openai` / `gpt-5-mini`). None of that traffic goes through this harness, so `memory_tokens_total: 0` means **unmeasured**, not free — the same shape letta has, and not comparable with a system whose zero is measured.

## Recall is budgeted in tokens, not in hits

`recall` takes `max_tokens`, not a `k`. How many results that buys depends on how long they are. The adapter asks for a deliberately generous budget (32k) so the budget is never what limits the count, then takes the first `k` the server ranked — which makes `k` comparable with the other systems. `--param max_tokens=N` drives the budget directly for anyone who wants to measure Hindsight on the axis its API is actually shaped around.

## Running it

```bash
docker compose -p hindsight-smoke -f benchmarks/hindsight/docker-compose.yaml \
  run --build --rm benchmark run --system hindsight \
  --dataset locomo --limit 1 --turns 40 --questions 5
docker compose -p hindsight-smoke -f benchmarks/hindsight/docker-compose.yaml down -v
```

The server needs an OpenAI key. It wants it as `HINDSIGHT_API_LLM_API_KEY`, which `env_file` can inject but cannot rename, and compose interpolation only sees an exported shell variable — so the server image's entrypoint maps `OPENAI_API_KEY` across. That works whether the key comes from `../../.env` locally or from the environment CI exports.

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `max_tokens` | `32000` | Recall's token budget. Lower it to measure Hindsight on its own budget axis. |
| `budget` | `mid` | Hindsight's own ranking effort; `high` spends more on recall. |
| `base_url` | `HINDSIGHT_BASE_URL`, else `http://localhost:8888` | Which server to measure. |
| `timeout` | `300` | Client timeout in seconds. |

## Notes

- **Isolation is a bank** (`conv-<id>`), Hindsight's own primitive and required on both `retain` and `recall`, so recall cannot cross conversations by construction. Teardown deletes exactly one bank.
- **Provenance is session-level.** A result is an extracted memory over a document, not a verbatim turn, so hits carry `session_ids` and never claim `turn_ids` — the same level as mem0 and graphiti. The document id is chosen at `retain` time rather than left to the server, so a hit names its session back without a lookup.
- **Half the results have no document of their own.** Recall returns `world` memories, which name the document they were extracted from, and `observation` memories — Hindsight's derived layer — which are synthesised across sources and carry `document_id`, `metadata` and `chunk_id` all `None`. Without a second hop those hits have no provenance, and the harness drops their questions rather than scoring 0.0: on a 94-question run that silently excluded 42 of them, including the entire `adversarial` category, and reported the surviving easier half as the result. `include_source_facts=True` exposes `source_fact_ids` that resolve through the response's `source_facts` map back to real documents, which keeps Hindsight's observation layer *in* the measurement instead of dropping it to make the numbers computable.
