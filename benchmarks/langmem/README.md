# langmem

Integration for [langchain-ai/langmem](https://github.com/langchain-ai/langmem). LangMem runs in-process and holds nothing itself: `create_memory_store_manager` reads and writes a LangGraph `BaseStore`, and the store is what keeps the memories and searches them. The one measured here is `PostgresStore` — Postgres with pgvector, the store LangMem's own docs reach for outside a notebook. Compose stands it up; there is no LangMem server.

The manager is used **standalone**, not inside a LangGraph agent, so direct mode stays a straight ingest/search measurement: a session's messages go to `manager.invoke`, which searches for related memories, extracts new ones with the configured model, and applies the inserts and updates.

## Provenance comes from the namespace

A LangMem memory is an extraction over a window of messages, and nothing on the stored value ties it back to a turn. But the manager writes under a **templated namespace**, and the store searches a namespace **prefix** — so:

- ingestion writes each session under `("memories", <conversation>, <session>)`, filled per call from the `configurable` config LangMem templates from;
- search asks for `("memories", <conversation>)`, which spans every session of that conversation while each hit still names the session it came from (`item.namespace[-1]`).

That is also what keeps recall inside one conversation, and what makes teardown a per-conversation delete that leaves the conversations running beside it under `--workers N` alone.

## Search is the store's, by LangMem's own design

`create_search_memory_tool` is a thin wrapper over `store.search(namespace, query=..., limit=...)`, so direct mode calls that same method with the same arguments rather than routing a tool call through an agent. LangMem's other search surface, `create_memory_searcher`, puts an LLM query-rewriting pass in front of it — query understanding rather than retrieval, and not comparable with the other systems here.

## One core fix came out of this integration

LangMem reaches OpenAI through langchain, and langchain-openai calls `client.with_raw_response.create(...)`. That hands back a `LegacyAPIResponse` — the HTTP response, not the parsed model — which has no `.usage`, so `OpenAIUsageTracker` counted the call and booked **zero tokens** against it. The first run of this integration reported `llm_calls: 2` with `llm_input_tokens: 0`; it now reports 1,887 input and 3,410 output tokens for the same work. The tracker reads through `parse()` when `.usage` is absent (`src/amb/callbacks/openai.py`); `parse()` is what the caller itself calls next and caches, so nothing is re-read or consumed twice. Any future langchain-based integration would have hit the same silent zero.

## Running it

```bash
docker compose -p langmem-smoke -f benchmarks/langmem/docker-compose.yaml \
  run --build --rm benchmark run --system langmem \
  --dataset locomo --limit 1 --turns 40 --questions 5
docker compose -p langmem-smoke -f benchmarks/langmem/docker-compose.yaml down -v
```

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `model` | `gpt-5-mini` | The extraction model, matching the rest of the comparison. Resolved by langchain's `init_chat_model`, so `provider:model` ids work too. |
| `embedding_model` | `text-embedding-3-small` | The store's embedder, at `embedding_dimensions` (1536 — also the pgvector column width, fixed at first `setup()`). |
| `query_limit` | `5` | LangMem's own default: how many existing memories the manager pulls in as context before extracting. It is what lets it update and consolidate rather than only append, and it is charged to ingestion. |
| (env) `LANGMEM_DATABASE_URL` | `postgresql://langmem:langmem@localhost:5432/langmem` | Where the store lives. |

## Notes

- **The store's migrations are its own.** `store.setup()` creates the `vector` extension and both tables, so the compose image is a stock `pgvector/pgvector` with no init script — PostgreSQL 17.11 + pgvector 0.8.6, pinned by digest because `pg17` moves and a different pgvector would change retrieval without changing the recorded LangMem version. They are check-then-create, so a process lock serializes them across `--workers N`.
- **Memories are unstructured strings.** LangMem also takes `schemas=[...]` pydantic models; the default (a plain string per memory) is what is measured, and `_content` handles both shapes so a schema experiment needs no adapter change.
- **`enable_deletes` stays at LangMem's default (off).** The manager updates and inserts; it does not retract. Turning it on changes what is being measured, not just how much it spends.
- **Agentic mode exposes LangMem's own two verbs**, with the bodies its `create_manage_memory_tool` / `create_search_memory_tool` give an agent: a `put` and a `search`. Only the `create` action is exposed — `update` and `delete` need a memory id the agent could only get from a search result, and ids are deliberately kept out of tool results because they are retrieval's scoring labels.
