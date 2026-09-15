# memos

Integration for [MemTensor/MemOS](https://github.com/MemTensor/MemOS) (PyPI `MemoryOS`). MemOS runs in-process: a `MOS` instance over one `GeneralMemCube` whose textual memory is the `tree_text` backend, which keeps its memories as a graph in Neo4j. Compose stands up the graph; there is no MemOS server.

The configuration is MemOS's own `get_default_config` / `get_default_cube_config`, with only the models, the endpoints and the ids injected — what is measured is MemOS's default stack, not this adapter's reading of it.

## Why `tree_text`

`MOS.add` runs MemOS's **MemReader** over a session's messages only for the `tree_text` backend: it extracts memories with the configured LLM, links them in the graph, and files them as `LongTermMemory` / `UserMemory`. Every other textual backend takes the same call and stores each message verbatim — that would measure a plain vector store wearing MemOS's name.

`tree_text` needs a graph. Neo4j Community has exactly one database and rejects `CREATE DATABASE`, so the adapter runs MemOS in its **shared-database tenant mode**: `use_multi_db=False` plus a `user_name` per conversation, which MemOS stores on every node and filters on in every query. Retrieval then uses `vector.similarity.cosine()` pre-filtering (Neo4j ≥ 5.18) rather than the global ANN index, which is also what keeps one conversation's top-k from being crowded out by another's.

## Three accommodations

1. **gpt-5-mini needs three request parameters changed.** MemOS builds every chat request with `temperature`, `top_p` and `max_tokens`, and OpenAI's reasoning models reject all three (`max_tokens` by name, in favour of `max_completion_tokens`). `_reasoning_llm_class` subclasses MemOS's own `OpenAILLM`, fixes the body it built, and is registered as the `openai` backend — same client, same parsing, same fallback; a non-reasoning model's body is left alone. `--param reasoning=false` opts out.
2. **MemOS's thread pools drop the ambient context.** Extraction, recall and embedding all run inside its `ContextThreadPoolExecutor`, which propagates only MemOS's own `RequestContext`. `OpenAIUsageTracker` keeps its counters in a `ContextVar`, so every token spent in a MemOS worker thread would be booked into a fresh context and lost — a published cost of zero this system never earned. `_propagate_context` makes `submit` carry the submitting thread's context (`map` goes through `submit`, so it is covered too).
3. **It writes its own diagnostics onto stdout**, which is where `amb run` writes the run summary. Its logger's console handler is configured there, and two of its graph methods `print()` the Cypher they run — parameters included, so 1536 floats of query embedding per search and per dedup lookup. `_quiet_logging` moves the logger to stderr (and its file handler off INFO, where the same dumps are hundreds of megabytes over a run). For the bare `print`s, `_quiet_graph_dumps` binds a `print` in `memos.graph_dbs.neo4j` that logs at debug instead. `print` resolves as a module global before it resolves as a builtin, so that reaches exactly the calls in the one module that makes them — `sys.stdout` is not replaced, `builtins` is not touched, and no other library's output changes. Nothing is discarded: `amb --log-level debug run` prints all of it. Measured at ~63 KB per question of embedding floats (~120 MB for one k-sweep cell), which is why the default level is where it stops. Importing MemOS also logs one record through its stdout handler before there is a config to patch, so stdlib logging is switched off across that single import (amb logs through loguru, which `logging.disable` does not reach).

`max_tokens` also moves from MemOS's default 1024 to 8192: a reasoning model can spend 1024 entirely on reasoning tokens and return no JSON for the extraction to parse.

## Two workspace-wide dependency overrides

MemoryOS 2.0.33 caps `openai<2` and `fastapi<0.116`, and one lockfile covers every member — so those caps would be the whole workspace's. Both are overridden in the root `pyproject.toml` (`[tool.uv] override-dependencies`): MemOS uses the parts of the openai SDK that did not change across the 2.0 line, and never imports FastAPI on this path (it is there for its API server). Its `transformers<5` does pull `huggingface-hub` back below 1.0 for every member — within `amb`'s own `>=0.34`, and the dataset loaders use `HfApi.list_repo_files` / `hf_hub_download`, unchanged across that line.

## Running it

```bash
docker compose -p memos-smoke -f benchmarks/memos/docker-compose.yaml \
  run --build --rm benchmark run --system memos \
  --dataset locomo --limit 1 --turns 40 --questions 5
docker compose -p memos-smoke -f benchmarks/memos/docker-compose.yaml down -v
```

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `model` | `gpt-5-mini` | The extraction model, matching the rest of the comparison. |
| `reasoning` | auto | Whether the model rejects `temperature` / `top_p` / `max_tokens`. Auto-detected from the model name; `false` restores MemOS's own request body. |
| `embedding_model` | `text-embedding-3-small` | Embedder, at `embedding_dimensions` (1536 — also the graph's vector dimension). |
| `search_mode` | `fast` | MemOS's own default: vector + graph recall, reranked locally. `fine` adds an LLM pass over the query, which is query understanding rather than retrieval and not comparable with the other systems. |
| `max_tokens` | `8192` | Ceiling per extraction call; see above for why MemOS's 1024 is not enough. |
| `reorganize` | `false` | MemOS's own default. `true` turns on its background graph reorganization. |
| (env) `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / `password` | Where the graph lives. |

## Notes

- **Isolation is a user plus a MemCube** (`conv<id>` / `cube<id>`), and node-level tenancy in the graph. MemOS derives its tenant tag from the user id by stripping `-` and `_`, so the adapter's slug is alphanumeric — two conversation ids cannot collapse onto one tenant.
- **One MemOS user per conversation.** A LoCoMo conversation has two speakers, but the unit of memory here is the conversation, so both speak as one user; the speaker survives in the text of every message and in the `user`/`assistant` roles.
- **Provenance is session-level.** A memory is an extraction over a window of a session's messages, not a verbatim turn, so hits carry `session_ids` and never claim `turn_ids`. MemOS keeps per-source snippets on a node, but its `simple_struct` reader does not carry a message id into them.
- **Teardown deletes one tenant's nodes** (`delete_all` is scoped to the cube's `user_name`), so the conversations running beside it under `--workers N` — nodes in the same database — are untouched. The Neo4j driver is closed with it: MemOS opens one per cube and exposes no `close`.
- **`.memos/` in the working directory** holds MemOS's SQLite user store and its own log file. In the container that is `/amb`, beside the mounted volumes.
