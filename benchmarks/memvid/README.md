# memvid

Integration for [memvid/memvid](https://github.com/memvid/memvid). memvid is a single-file memory: content, a BM25 index (tantivy) and an HNSW vector index packed into one `.mv2`. The Rust core ships inside the `memvid-sdk` wheel, so it runs in-process and there is no server for compose to stand up. One `.mv2` per conversation under `MEMVID_ROOT`, deleted at teardown.

Every turn is one frame, stored verbatim with its session timestamp. There is no LLM in the write path — memvid's own `enrich` is a rule-based pass, off by default — so the ingestion model is none and the only spend is the embedder: memvid's `OpenAIEmbeddings` (`text-embedding-3-small`) runs in-process through the openai SDK, where `OpenAIUsageTracker` sees it. Spend **is** measured, not assumed.

## Search modes

`find` takes a mode, and the three behave differently enough to matter (measured on 2.0.160):

| `mode` | What it does |
| --- | --- |
| `lex` | tantivy BM25 with **every query term required**. A natural-language question almost always contains a word no turn has, and then returns nothing. |
| `sem` | Pure vector similarity, `k` nearest frames. |
| `hybrid` | The lexical match set re-ranked by reciprocal-rank fusion with the vector list; when the lexical match set is empty it falls back to `sem`. |

The default follows the SDK's own auto mode: `sem` when the file has vectors, `lex` when `embedding_model=none`. `--param mode=hybrid` measures the fusion.

## Two SDK conveniences the adapter turns off

Both read `OPENAI_API_KEY` from the environment and spend through the Rust core's own HTTP client — spend no tracker can see, which would publish a false zero:

- **Writes embed natively whenever the key is set**, regardless of the documented `enable_embedding` default. The adapter embeds in Python and hands `put_many` the vectors with `enable_embedding` explicitly off. The `put_many(embedder=)` chunk path is avoided as well: in 2.0.160 its vectors never reach the index (`vec_index_bytes` stays 0 and `sem` raises MV011).
- **`find`'s auto mode embeds the query natively.** The adapter always passes the mode and the query vector itself.

Verified with a bogus key in the environment: every call the adapter makes stays offline; the SDK defaults (`put`, `find` with no mode) reach OpenAI and fail with a 401.

## Telemetry

The SDK reports every `put`/`find` to memvid.com unless `MEMVID_TELEMETRY=0`. The compose file sets it; set it yourself for a bare `amb run`.

## Running it

```bash
docker compose -p memvid-smoke -f benchmarks/memvid/docker-compose.yaml \
  run --build --rm benchmark run --system memvid \
  --dataset locomo --limit 1 --turns 40 --questions 5
```

## Parameters

| `--param` | Default | What it changes |
| --- | --- | --- |
| `embedding_model` | `text-embedding-3-small` | The embedder; `none` writes a lexical-only file. |
| `mode` | auto | `lex`, `sem` or `hybrid`; auto is `sem` with an embedder, `lex` without. |
| (env) `MEMVID_ROOT` | `.memvid` | Where the per-conversation `.mv2` files live. |

## Notes

- **Provenance is turn-level.** Frames are verbatim turns, so hits carry both `turn_ids` and `session_ids`. A hit returns the frame's uri and a snippet padded with its title and metadata, so a per-conversation uri map restores the verbatim turn and its provenance. The uri is spelled in letters only, because it is indexed with the text and a digit in it could match a question. `put_many` returns sequence numbers, not frame ids — the uri is the only handle that round-trips. Lexical results can list a frame twice and are de-duplicated.
- **Batches per session.** One `put_many` (and one embeddings call) per dataset session; a query with no words at all is answered empty without a call, because the lexical parser rejects it.
- **One thread in the handle at a time.** The core handle is a single-borrow PyO3 object and fails with `Already borrowed` under concurrent calls, which agentic mode produces (pydantic-ai runs parallel tool calls on worker threads). The adapter serializes every core call per instance; embedding happens outside the lock.
- **`create` truncates**, so a file left behind by an aborted run never seeds the next one.
- **Version.** `system_version` is the `memvid-sdk` wheel, which bundles the core; the core's own version (2.0.140 in 2.0.160) is recorded in the run's stats as `core_version`.
