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

"""memvid (memvid/memvid) — single-file memory: BM25 + HNSW vectors in one .mv2.

Runs in-process: the Rust core ships inside the ``memvid-sdk`` wheel, so
there is no server for compose to stand up. Each conversation is one
``.mv2`` file under ``MEMVID_ROOT`` — the isolation boundary, and the
teardown unit (close, delete).

Every turn is one frame, stored verbatim. There is no LLM in the write
path (memvid's own ``enrich`` is rule-based and off by default), so the
ingestion model is None and the only spend is the embedder: memvid's
``OpenAIEmbeddings`` (text-embedding-3-small) runs in-process through
the openai SDK, where the OpenAIUsageTracker sees it. ``--param
embedding_model=none`` drops it for a lexical-only file.

Two SDK conveniences are overridden on purpose. Both read OPENAI_API_KEY
from the environment and spend through the Rust core's own HTTP client,
which no tracker sees:

* A write embeds natively whenever the key is set. Vectors are computed
  in Python and handed to ``put_many`` with ``enable_embedding`` off.
  ``put_many(embedder=)`` — the chunk path — is avoided as well: in
  2.0.160 its vectors never reach the index and semantic search raises
  MV011.
* ``find``'s auto mode embeds the query natively. The adapter resolves
  the mode the same way the SDK does (semantic with vectors, lexical
  without), passes it explicitly, and supplies the query vector itself.

Hits carry the frame's uri and a snippet padded with its title and
metadata; a per-conversation uri map restores the verbatim turn and its
provenance. (`put_many` returns sequence numbers, not frame ids, so the
uri is the only handle that round-trips.)
"""

import os
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

if TYPE_CHECKING:
    from memvid_sdk import Memvid
    from memvid_sdk.embeddings import EmbeddingProvider

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_ROOT = ".memvid"
MODES = ("lex", "sem", "hybrid")
_PATH_SAFE = re.compile(r"[^a-zA-Z0-9_.@+-]+")


def _uri(index: int) -> str:
    """The frame's uri, spelled in letters no question contains.

    The uri is indexed with the text, so a digit in it could match a
    question lexically.
    """
    letters = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        letters = chr(97 + rest) + letters
    return f"zz{letters}"


class MemvidMemory(Memory):
    """memvid: verbatim turns in one .mv2 per conversation, BM25 and/or vectors."""

    name: ClassVar[str] = "memvid"
    description: ClassVar[str] = "memvid — single-file memory, BM25 + HNSW in one .mv2"
    sdk_dist: ClassVar[str | None] = "memvid-sdk"

    def __init__(
        self,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        mode: str | None = None,
        **params: object,
    ) -> None:
        """Pin the embedder and the search mode memvid will use.

        Raises:
            ValueError: For an unknown mode, or a vector mode with no embedder.
        """
        super().__init__(**params)
        self.root = Path(os.environ.get("MEMVID_ROOT", DEFAULT_ROOT))
        self.embedding_model = embedding_model
        # None resolves like the SDK's auto mode: semantic when vectors exist
        self.mode = mode or ("sem" if embedding_model else "lex")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        if self.mode != "lex" and not embedding_model:
            raise ValueError(f"mode={self.mode!r} needs an embedding model")
        self._embedder: EmbeddingProvider | None = None
        # The core handle is a single-borrow PyO3 object: two threads in it at
        # once fail with "Already borrowed", and agentic mode runs parallel
        # tool calls on worker threads. Embedding happens outside the lock.
        self._lock = threading.Lock()
        self._handles: dict[str, Memvid] = {}
        # uri -> (text, turn_ids, session_id): hits return only the uri and a
        # padded snippet
        self._frames: dict[str, dict[str, tuple[str, list[str], str]]] = {}

    def setup(self) -> None:
        """Build the embedder, when one is configured, and the file root."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.embedding_model:
            from memvid_sdk.embeddings import OpenAIEmbeddings

            self._embedder = OpenAIEmbeddings(model=self.embedding_model)

    def _path(self, conversation_id: str) -> Path:
        return self.root / f"{_PATH_SAFE.sub('-', conversation_id)}.mv2"

    def _handle(self, conversation_id: str) -> "Memvid":
        """The conversation's open file, created on first use.

        `create` truncates, so a file left behind by an aborted run never
        seeds this one.
        """
        handle = self._handles.get(conversation_id)
        if handle is None:
            import memvid_sdk

            handle = memvid_sdk.create(
                str(self._path(conversation_id)),
                enable_vec=self._embedder is not None,
            )
            self._handles[conversation_id] = handle
        return handle

    def store(
        self,
        conversation_id: str,
        requests: list[dict],
        *,
        session_id: str,
        turn_ids: list[list[str]],
    ) -> None:
        """Put one batch of frames (`put_many` requests) with their provenance.

        One embeddings call covers the batch; the vectors travel with the
        request and native embedding stays off, so the core never spends
        on its own.
        """
        embeddings = None
        identity = None
        if self._embedder is not None:
            embeddings = self._embedder.embed_documents([r["text"] for r in requests])
            identity = {
                "provider": "openai",
                "model": self._embedder.model_name,
                "dimension": self._embedder.dimension,
            }
        with self._lock:
            handle = self._handle(conversation_id)
            frames = self._frames.setdefault(conversation_id, {})
            requests = [
                {**request, "uri": _uri(len(frames) + index)}
                for index, request in enumerate(requests)
            ]
            try:
                handle.put_many(
                    requests,
                    embeddings=embeddings,
                    embedding_identity=identity,
                    opts={"enable_embedding": False},
                )
            except Exception:
                # the benchmark log outlives the file; keep the batch diagnosable
                logger.bind(scope="memvid").error(
                    "put_many failed: conversation={} session={} frames={}",
                    conversation_id,
                    session_id,
                    len(requests),
                )
                raise
            for request, ids in zip(requests, turn_ids, strict=True):
                frames[request["uri"]] = (request["text"], list(ids), session_id)

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Store every turn of the session as one frame, in one batch."""
        if not session.turns:
            return
        requests = []
        for turn in session.turns:
            text = f"{turn.speaker}: {turn.text}"
            if session.timestamp:
                text = f"({session.timestamp}) {text}"
            requests.append({"title": turn.speaker, "label": "turn", "text": text})
        self.store(
            conversation_id,
            requests,
            session_id=session.session_id,
            turn_ids=[[turn.turn_id] for turn in session.turns],
        )

    def find_hits(
        self, conversation_id: str, query: str, k: int = 10, mode: str | None = None
    ) -> list[MemoryHit]:
        """Find inside the conversation's file, with provenance restored.

        `mode` overrides the configured one for a single call (agentic
        mode lets the agent pick); it must have vectors to lean on.

        Raises:
            ValueError: For a vector mode when no embedder is configured.
        """
        mode = mode or self.mode
        # the lexical parser rejects an empty query outright
        if not query.strip():
            return []
        query_embedding = None
        if mode != "lex":
            if self._embedder is None:
                raise ValueError(f"mode={mode!r} needs an embedding model")
            query_embedding = self._embedder.embed_query(query)
        with self._lock:
            handle = self._handles.get(conversation_id)
            if handle is None:
                return []
            try:
                result = handle.find(
                    query, k=k, mode=mode, query_embedding=query_embedding
                )
            except Exception:
                # the lexical parser has rejected live queries (a stray `*`);
                # log the exact input so the row's error is diagnosable
                logger.bind(scope="memvid").error(
                    "find failed: conversation={} mode={} query={!r}",
                    conversation_id,
                    mode,
                    query,
                )
                raise
            frames = dict(self._frames.get(conversation_id, {}))
        hits: list[MemoryHit] = []
        seen: set[str] = set()
        for hit in result["hits"]:
            uri = str(hit.get("uri", ""))
            # lexical results can list a frame twice
            if uri in seen:
                continue
            seen.add(uri)
            text, turn_ids, session_id = frames.get(
                uri, (str(hit.get("snippet", "")), [], "")
            )
            hits.append(
                MemoryHit(
                    content=text,
                    score=hit.get("score"),
                    turn_ids=turn_ids,
                    session_ids=[session_id] if session_id else [],
                    metadata={"frame_id": hit.get("frame_id"), "uri": uri},
                )
            )
        return hits[:k]

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return the k best frames for the query, inside the conversation."""
        return self.find_hits(conversation_id, query, k=k)

    def teardown(self) -> None:
        """Close and delete this sample's files, and only this sample's.

        Scoped on purpose: under `--workers N` the other conversations'
        files are siblings under the same root.
        """
        with self._lock:
            for conversation_id, handle in self._handles.items():
                handle.close()
                self._path(conversation_id).unlink(missing_ok=True)
            self._handles.clear()
            self._frames.clear()

    def stats(self) -> dict:
        """Report what this run stored, how big the files are, and the core."""
        import memvid_sdk

        native = memvid_sdk.info().get("native") or {}
        with self._lock:
            return {
                "stored_frames": sum(len(f) for f in self._frames.values()),
                "file_bytes": sum(
                    int(h.stats().get("size_bytes", 0)) for h in self._handles.values()
                ),
                "mode": self.mode,
                "core_version": native.get("memvid_core_version"),
            }
