# MIT License
#
# Copyright (c) 2026 René-Jean Corneille
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Cognee (topoteretes/cognee) — embedded knowledge-graph memory.

Workspace package ``cognee-benchmark`` (this directory). cognee can run
fully embedded (LanceDB, ladybug and sqlite on local disk), but this
integration’s docker-compose stack benchmarks it against Postgres (with
pgvector) for the relational + vector stores while keeping the graph on
cognee’s embedded ladybug backend.
The providers are chosen entirely by environment variable, so an
embedded run is still a compose file away.

Ingestion is cognee's own: ``add`` stages a session's turns and
``cognify`` builds the graph from them, extracting with the model pinned
below (gpt-5-mini by default, the same model every other system in the
comparison ingests with) and embedding with text-embedding-3-small
rather than cognee's own 3072-dimension default.

Each conversation is a cognee *dataset*. With backend access control on
— the default for this backend pair — a dataset owns its own graph and
vector files, so recall cannot cross conversations by construction
rather than by filter.

Two properties of cognee shape this adapter and are not obvious:

* It keeps module-level ``asyncio.Lock`` singletons. A lock binds to the
  first event loop that contends on it, so one loop per worker thread
  raises "bound to a different event loop" — and the threads that do not
  raise then hang. Everything is therefore marshalled onto ONE
  process-wide loop through ``_await``.
* A cold start does select-then-create on a default user with no retry,
  so concurrent first calls collide on ``UNIQUE constraint failed:
  users.email``. ``setup`` runs that warmup once per process, serially,
  before workers fan out.
"""

import asyncio
import os
import threading
from contextvars import ContextVar
from typing import Any, ClassVar

from amb.base import Memory
from amb.constants import TOKEN_TRACKING_KEYS
from amb.contracts import MemoryHit, Session
from amb.logs import logger

DEFAULT_INGESTION_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# text-embedding-3-small's native width, and the same width every other
# system in the comparison embeds at
DEFAULT_EMBEDDING_DIMENSIONS = 1536
# what a recall returns: CHUNKS is the raw retrieved text, the analogue of
# every other system's search. The *_COMPLETION types run another LLM pass
# and return a written answer, which is answer generation, not retrieval.
DEFAULT_SEARCH_TYPE = "CHUNKS"

# One process-wide loop, running in its own thread; see the module
# docstring for why per-thread loops cannot work here.
_LOOP: asyncio.AbstractEventLoop | None = None
_GUARD = threading.Lock()
# a separate lock, held for the whole of the one-time schema warmup so
# later workers *wait* for it rather than racing past it. It cannot be
# _GUARD: the warmup awaits, and `_await` takes _GUARD to reach the loop.
_WARM_LOCK = threading.Lock()
_WARMED = False
_USAGE_INSTALLED = False


def _loop() -> asyncio.AbstractEventLoop:
    """The shared loop, started on first use."""
    global _LOOP
    with _GUARD:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_LOOP.run_forever, name="cognee-loop", daemon=True
            ).start()
    return _LOOP


# The counters of the conversation whose work is currently running.
# `run_coroutine_threadsafe` copies the submitting thread's context, so a
# value set here in a worker thread reaches the litellm callback that
# fires on the shared loop — which is what lets concurrent conversations
# book their own spend instead of one global total.
_COUNTERS: ContextVar[dict | None] = ContextVar("cognee_counters", default=None)


def _await(coro: Any, counters: dict | None = None, timeout: float = 1800) -> Any:
    """Run a coroutine on the shared loop, billing it to `counters`."""
    token = _COUNTERS.set(counters)
    try:
        return asyncio.run_coroutine_threadsafe(coro, _loop()).result(timeout)
    finally:
        _COUNTERS.reset(token)


def _install_usage_logger() -> None:
    """Book every litellm call against the conversation that caused it.

    cognee reaches OpenAI through litellm, not through the openai SDK
    methods `OpenAIUsageTracker` patches, so without this a cognee run
    reports zero tokens — and cognee is the most expensive system per
    session in the comparison.
    """
    global _USAGE_INSTALLED
    with _GUARD:
        if _USAGE_INSTALLED:
            return
        _USAGE_INSTALLED = True

    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    def _book(kwargs: dict, response_obj: Any) -> None:
        counters = _COUNTERS.get()
        if counters is None:
            return
        usage = getattr(response_obj, "usage", None)
        if usage is None and isinstance(response_obj, dict):
            usage = response_obj.get("usage")
        if usage is None:
            return
        get = (
            usage.get
            if isinstance(usage, dict)
            else lambda k, d=0: getattr(usage, k, d)
        )  # noqa: E731
        # embeddings report prompt tokens only; completions report both.
        # `in`, not `startswith`: litellm names the async variants
        # `aembedding`/`acompletion`, so a prefix test books every async
        # embedding as an LLM call — and the two are priced an order of
        # magnitude apart
        embedding = "embedding" in str(kwargs.get("call_type", "")).lower()
        if embedding:
            counters["embedding_calls"] += 1
            counters["embedding_tokens"] += get("prompt_tokens", 0) or 0
        else:
            counters["llm_calls"] += 1
            counters["llm_input_tokens"] += get("prompt_tokens", 0) or 0
            counters["llm_output_tokens"] += get("completion_tokens", 0) or 0

    class _UsageLogger(CustomLogger):
        """litellm's success hook, in both its sync and async forms."""

        def log_success_event(
            self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _book(kwargs, response_obj)

        async def async_log_success_event(
            self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
        ) -> None:
            _book(kwargs, response_obj)

    litellm.callbacks.append(_UsageLogger())


class CogneeMemory(Memory):
    """Cognee: knowledge graph built by its own extraction pipeline."""

    name: ClassVar[str] = "cognee"
    description: ClassVar[str] = "Cognee — embedded knowledge-graph memory"
    sdk_dist: ClassVar[str | None] = "cognee"

    def __init__(
        self,
        model: str | None = DEFAULT_INGESTION_MODEL,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        embedding_dimensions: int | str = DEFAULT_EMBEDDING_DIMENSIONS,
        search_type: str = DEFAULT_SEARCH_TYPE,
        **params: object,
    ) -> None:
        """Pin the models and the retrieval type cognee will use."""
        super().__init__(**params)
        self.model = model
        self.embedding_model = embedding_model
        # --param values arrive as strings
        self.embedding_dimensions = int(embedding_dimensions)
        self.search_type = search_type
        # one add() is one cognee document, and every chunk a search
        # returns carries its document_id — so mapping that id to the
        # session restores provenance exactly, with no text matching
        self._documents: dict[str, str] = {}
        # the conversation this instance was built for; teardown needs it
        # and the base contract does not pass it
        self._conversation_id: str | None = None
        self._dataset: Any = None
        self._user: Any = None
        # this conversation's own spend, filled by the litellm callback
        # and read by TiktokenUsageTracker at each lifecycle boundary
        self._usage: dict[str, int] = dict.fromkeys(TOKEN_TRACKING_KEYS, 0)

    def usage_counters(self) -> dict:
        """This instance's token spend so far (copy)."""
        return dict(self._usage)

    def models(self) -> dict[str, str | None]:
        """The models cognee calls internally; recorded in every run.

        The keys are the contract's (`ingestion_model` / `embedding_model`)
        — the Runner reads them by those exact names. Returning
        "ingestion"/"embedding" silently dropped both from every run's
        identity, so runs against different extraction models were
        indistinguishable in the results.
        """
        return {
            "ingestion_model": self.model,
            "embedding_model": self.embedding_model,
        }

    @staticmethod
    def _configure_env() -> None:
        """Set cognee's environment before it is imported.

        Its config objects are cached singletons built on first access
        during import, so these have no effect if set afterwards. The
        names are unprefixed and the paths must be absolute. cognee does
        not read OPENAI_API_KEY — it wants LLM_API_KEY.
        """
        root = os.environ.get("COGNEE_ROOT", "/amb/.cognee")
        os.environ.setdefault("DATA_ROOT_DIRECTORY", f"{root}/data")
        os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", f"{root}/system")
        os.environ.setdefault("CACHE_ROOT_DIRECTORY", f"{root}/cache")
        os.environ.setdefault("COGNEE_LOGS_DIR", f"{root}/logs")
        if key := os.environ.get("OPENAI_API_KEY"):
            os.environ.setdefault("LLM_API_KEY", key)
        # otherwise cognee spends one live probe call per process
        os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

    def _pin_models(self) -> None:
        """Override cognee's defaults, including its 3072-dim embedder.

        These setters run after import deliberately: cognee loads a
        `.env` with override, so a repo-root file would otherwise win on
        local runs.
        """
        import cognee

        if self.model:
            cognee.config.set_llm_provider("openai")
            cognee.config.set_llm_model(f"openai/{self.model}")
        if self.embedding_model:
            cognee.config.set_embedding_provider("openai")
            cognee.config.set_embedding_model(f"openai/{self.embedding_model}")
            # the dimension does not follow the model: cognee keeps the
            # 3072 of its own text-embedding-3-large default, which
            # text-embedding-3-small rejects outright (400, "must be less
            # than or equal to 1536")
            cognee.config.set_embedding_dimensions(self.embedding_dimensions)

    def setup(self) -> None:
        """Import cognee, pin its models, and warm the shared state once."""
        global _WARMED
        self._configure_env()
        from cognee.modules.users.methods import get_default_user

        self._pin_models()
        _install_usage_logger()
        # Serial, once per process. The lock is held across the await, so a
        # worker arriving mid-warmup blocks until the schema exists rather
        # than racing on to get_default_user() and hitting
        # DatabaseNotCreatedError — and concurrent cold starts cannot
        # collide on the default user's unique constraint.
        with _WARM_LOCK:
            if not _WARMED:
                from cognee.modules.engine.operations.setup import (
                    setup as cognee_setup,
                )

                _await(cognee_setup())
                _await(get_default_user())
                _WARMED = True
        self._user = _await(get_default_user())

    def store(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str] | None = None,
    ) -> None:
        """Add one piece of content to the conversation's dataset.

        `node_set` tags it with the session and conversation, which
        survives into every chunk's payload — a second provenance channel
        beside the document id, and the analogue of the forced anchors
        the other integrations write.
        """
        import cognee

        self._conversation_id = conversation_id
        _await(
            cognee.add(
                content,
                dataset_name=conversation_id,
                node_set=[session_id, f"conv:{conversation_id}"],
            ),
            self._usage,
        )
        # one add is one data item, and the id of that item is exactly the
        # `document_id` every chunk carries back from a search — so the
        # items this dataset has gained since the last call are this
        # session's, and that is the primary provenance channel
        self._claim_documents(conversation_id, session_id)
        _await(cognee.cognify(datasets=[conversation_id]), self._usage)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Stage the session in its conversation's dataset, then cognify it."""
        if not session.turns:
            return
        self.store(
            conversation_id,
            "\n".join(f"{turn.speaker}: {turn.text}" for turn in session.turns),
            session_id=session.session_id,
            turn_ids=[turn.turn_id for turn in session.turns],
        )

    def _claim_documents(self, conversation_id: str, session_id: str) -> None:
        """Attribute this dataset's not-yet-seen data items to a session."""
        import cognee

        dataset_id = self._dataset_id(conversation_id)
        if dataset_id is None:
            return
        for item in _await(cognee.datasets.list_data(dataset_id, self._user)) or []:
            item_id = str(getattr(item, "id", "") or "")
            if item_id:
                self._documents.setdefault(item_id, session_id)

    def _dataset_id(self, conversation_id: str) -> Any:
        """This conversation's dataset id, looked up once."""
        import cognee

        if self._dataset is None:
            for dataset in _await(cognee.datasets.list_datasets(self._user)) or []:
                if getattr(dataset, "name", None) == conversation_id:
                    self._dataset = dataset.id
                    break
        return self._dataset

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k retrieved chunks from this conversation's dataset."""
        if not query.strip():
            return []
        import cognee
        from cognee.modules.search.types import SearchType

        try:
            results = _await(
                cognee.search(
                    query_text=query,
                    query_type=SearchType[self.search_type],
                    datasets=[conversation_id],
                    top_k=k,
                ),
                self._usage,
            )
        except Exception:
            # the payload is worth having in the benchmark log, which
            # outlives the container
            logger.bind(scope="cognee").error(
                "search failed: dataset={!r} type={!r} query={!r}",
                conversation_id,
                self.search_type,
                query,
            )
            raise
        hits = []
        # one bundle per dataset queried, each holding the chunks
        for bundle in results or []:
            for chunk in bundle.get("search_result", []) or []:
                document_id = str(chunk.get("document_id", ""))
                session_id = self._documents.get(document_id)
                hits.append(
                    MemoryHit(
                        content=chunk.get("text", ""),
                        session_ids=[session_id] if session_id else [],
                        metadata={
                            "chunk_id": str(chunk.get("id", "")),
                            "document_name": chunk.get("document_name"),
                        },
                    )
                )
        return hits[:k]

    def teardown(self) -> None:
        """Empty this conversation's dataset, and only this one.

        Never `prune_system()`/`prune_data()` here: those are global and
        would tear down the datasets of the conversations running beside
        this one under `--workers N`. `empty_dataset` is scoped to one id.
        """
        if self._conversation_id is None:
            return
        try:
            import cognee

            datasets = _await(cognee.datasets.list_datasets(self._user))
            for dataset in datasets or []:
                if getattr(dataset, "name", None) == self._conversation_id:
                    _await(cognee.datasets.empty_dataset(dataset.id, self._user))
                    break
        except Exception:  # noqa: BLE001 - teardown must not fail a scored run
            logger.bind(scope="cognee").warning(
                "could not empty dataset {!r}", self._conversation_id
            )
        finally:
            self._documents.clear()

    def stats(self) -> dict:
        """Report what this run stored, and how it retrieved."""
        return {
            "documents": len(self._documents),
            "search_type": self.search_type,
        }
