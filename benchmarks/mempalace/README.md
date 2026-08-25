# mempalace

Integration for [MemPalace/mempalace](https://github.com/MemPalace/mempalace).

## Embedder: matched to the comparison, not local by default

MemPalace ships two on-device embedders (`minilm` = all-MiniLM-L6-v2, 384-dim; `embeddinggemma`) and also speaks an OpenAI-compatible `/v1/embeddings` endpoint under the reserved model name `openai-compat`.

This integration defaults to **`text-embedding-3-small`**, so the embedder matches every other system in the comparison rather than being the one axis MemPalace differs on. `--param embedding_model=minilm` restores the local, zero-spend configuration.

**That choice changes what the token column means.** MemPalace fetches embeddings over stdlib `urllib`, not the openai SDK, so `OpenAIUsageTracker` cannot see the spend: on the API embedder its reported `0` tokens is **false**. Only the local embedders make that zero true. `stats()` records `embedder_is_local` so a run says which it was.

There is still **no ingestion LLM** either way — MemPalace stores verbatim and never summarises, extracts or paraphrases. That part is by design and not configurable.
