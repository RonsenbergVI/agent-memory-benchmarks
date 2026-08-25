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

"""MemPalace (MemPalace/mempalace) — local-first verbatim memory.

Workspace package ``mempalace-benchmark`` (this directory); the member
cannot be ``mempalace``, which is the SDK's own distribution name.

MemPalace stores conversation text verbatim as *drawers* and retrieves
with cosine similarity re-ranked by BM25. It runs no LLM: nothing is
summarised, extracted or paraphrased, so there is no ingestion model.
The embedder defaults to text-embedding-3-small to match the rest of the
comparison; ``--param embedding_model=minilm`` restores MemPalace's own
local one, which is the only configuration whose zero token spend is
real rather than merely unobserved.

Ingestion is MemPalace's own ``mine``: each session is written to a file
under ``<root>/<conversation>/sessions/`` and mined into that
conversation's palace. Isolation is a palace directory per conversation
rather than a filter, which also keeps MemPalace's per-palace write lock
from serialising ``--workers N``.

Provenance is exact at both levels. A hit names the ``source_file`` it
was cut from, and storage is verbatim with one turn per line, so a turn
was retrieved exactly when its whole line comes back in the hit's text —
which holds even though chunks cut mid-line and a hit is a drawer
stitched to its neighbours.
"""

import os
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from amb.base import Memory
from amb.contracts import MemoryHit, Session
from amb.logs import logger

# matched to the rest of the comparison; `--param embedding_model=minilm`
# restores MemPalace's own local embedder
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# run on-device; anything else goes to `openai-compat` over HTTP
LOCAL_EMBEDDERS = ("minilm", "embeddinggemma")
DEFAULT_EMBEDDING_API_URL = "https://api.openai.com/v1"
# MemPalace's own default; "union" also pulls in lexical candidates
DEFAULT_CANDIDATE_STRATEGY = "vector"
# relative, so it lands under the container's WORKDIR; MEMPALACE_ROOT moves it
DEFAULT_ROOT = ".mempalace"
# the room MemPalace routes to when no mempalace.yaml declares others
DEFAULT_ROOM = "general"
# what an agentic write is filed as, in place of a mined file's path
AGENT_SOURCE = "agent://{session_id}"

# the local embedder unpacks its ONNX model on first use into a shared
# cache; one warmup per process so concurrent mines cannot race it
_WARM_LOCK = threading.Lock()
_WARMED = False
# its own lock, not _WARM_LOCK: the warmup holds that one across an await
# on the embedder, and these two must never be able to wait on each other
_MUTE_LOCK = threading.Lock()
_MUTED_STDOUT: "_ThreadMutedStdout | None" = None


class _ThreadMutedStdout:
    """A stdout that drops the writes of threads which asked to be quiet.

    MemPalace's miner narrates its progress to stdout, and that stream
    belongs to the harness — its progress bars and the run summary it
    prints at the end. `contextlib.redirect_stdout` cannot silence the
    miner here: it swaps `sys.stdout` process-wide, so under `--workers N`
    one conversation's redirect also silences every other thread, and
    leaving the block closes the file the others were left pointing at
    ("I/O operation on closed file", raised anywhere, including in the
    CLI after the run). This proxy is installed once and mutes per thread
    instead, which is the granularity the problem actually has.
    """

    def __init__(self, stream: Any) -> None:
        """Wrap the real stdout, with nothing muted yet."""
        self._stream = stream
        self._local = threading.local()

    @contextmanager
    def muted(self) -> Iterator[None]:
        """Drop this thread's writes for the duration of the block."""
        self._local.muted = True
        try:
            yield
        finally:
            self._local.muted = False

    def write(self, data: str) -> int:
        """Write through, unless this thread is muted."""
        if getattr(self._local, "muted", False):
            return len(data)
        return int(self._stream.write(data))

    def flush(self) -> None:
        """Flush the wrapped stream."""
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        """Everything else — `encoding`, `isatty`, ... — is the real stream's."""
        return getattr(self._stream, name)


def _mute() -> _ThreadMutedStdout:
    """Install the per-thread stdout mute once, and return it."""
    global _MUTED_STDOUT
    with _MUTE_LOCK:
        if _MUTED_STDOUT is None:
            _MUTED_STDOUT = _ThreadMutedStdout(sys.stdout)
            sys.stdout = _MUTED_STDOUT
    return _MUTED_STDOUT


@dataclass
class _StagedSession:
    """One mined session file, and what each of its lines attests."""

    session_id: str
    # (rendered line, turn_id) per turn, in transcript order
    lines: list[tuple[str, str]] = field(default_factory=list)

    def turns_within(self, text: str) -> list[str]:
        """The turns this text holds whole, in transcript order."""
        held = set(text.split("\n"))
        return [turn_id for line, turn_id in self.lines if line in held]


class MemPalaceMemory(Memory):
    """MemPalace: verbatim drawers, local embeddings, hybrid retrieval."""

    name: ClassVar[str] = "mempalace"
    description: ClassVar[str] = "MemPalace — local-first verbatim memory"
    sdk_dist: ClassVar[str | None] = "mempalace"

    def __init__(
        self,
        root: str | None = None,
        embedding_model: str | None = DEFAULT_EMBEDDING_MODEL,
        candidate_strategy: str = DEFAULT_CANDIDATE_STRATEGY,
        **params: object,
    ) -> None:
        """Pin the local embedder and the retrieval strategy MemPalace uses."""
        super().__init__(**params)
        # `mine` resolves its project dir then calls `relative_to` on each
        # file, so a relative root raises "is not in the subpath of"
        self.root = (
            Path(root or os.environ.get("MEMPALACE_ROOT", DEFAULT_ROOT))
            .expanduser()
            .resolve()
        )
        # there is no extraction LLM: `models()` reads this and reports None
        self.model: str | None = None
        self.embedding_model = embedding_model
        self.candidate_strategy = candidate_strategy
        # the conversation this instance was built for; teardown needs it and
        # the base contract does not pass it
        self._conversation_id: str | None = None
        # file name -> the session it was written from. A hit names the file
        # it was cut from, so this is the primary provenance channel.
        self._staged: dict[str, _StagedSession] = {}
        # drawer id -> (session_id, turn_ids) for agentic writes, which are
        # filed directly and have no session file to locate text inside
        self._written: dict[str, tuple[str, list[str]]] = {}

    def models(self) -> dict[str, str | None]:
        """The models MemPalace uses; the ingestion one is genuinely None."""
        return {"ingestion_model": None, "embedding_model": self.embedding_model}

    def _configure_env(self) -> None:
        """Set MemPalace's environment before anything imports it.

        Its config is read through cached singletons, and chromadb builds
        its telemetry client at import, so both have to be in place first.
        """
        if self.embedding_model in LOCAL_EMBEDDERS:
            os.environ.setdefault("MEMPALACE_EMBEDDING_MODEL", self.embedding_model)
        elif self.embedding_model:
            # "openai-compat" is MemPalace's reserved name for an
            # OpenAI-compatible endpoint. It fetches over stdlib urllib, not
            # the openai SDK, so this spend is invisible to the tracker —
            # a run configured this way reports a false zero.
            os.environ.setdefault("MEMPALACE_EMBEDDING_MODEL", "openai-compat")
            os.environ.setdefault("MEMPALACE_EMBEDDING_API_MODEL", self.embedding_model)
            os.environ.setdefault(
                "MEMPALACE_EMBEDDING_API_URL",
                os.environ.get("OPENAI_BASE_URL", DEFAULT_EMBEDDING_API_URL),
            )
            if key := os.environ.get("OPENAI_API_KEY"):
                os.environ.setdefault("MEMPALACE_EMBEDDING_API_KEY", key)
        # chromadb ships posthog telemetry on by default; a benchmark run
        # has no business phoning home about the corpus it is measuring
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    @staticmethod
    def _warm_embedder() -> None:
        """Load the local embedder once per process, before workers fan out."""
        global _WARMED
        with _WARM_LOCK:
            if _WARMED:
                return
            try:
                from mempalace.embedding import get_embedding_function

                get_embedding_function()(["warmup"])
            except Exception:  # noqa: BLE001 - an optimisation, not a step
                # the model is baked into the image, so this only ever fires
                # on a cold local run; the real mine raises the useful error
                logger.bind(scope="mempalace").warning(
                    "could not warm the embedder; the first mine will load it"
                )
            _WARMED = True

    def setup(self) -> None:
        """Configure MemPalace and load its embedder once per process."""
        self._configure_env()
        self.root.mkdir(parents=True, exist_ok=True)
        _mute()
        self._warm_embedder()

    @staticmethod
    def _slug(value: str) -> str:
        """A file- and wing-safe token for an arbitrary dataset id."""
        slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in value.lower())
        return slug.strip("-") or "unnamed"

    def _conversation_root(self, conversation_id: str) -> Path:
        """Everything one conversation owns: its palace and its session files."""
        return self.root / self._slug(conversation_id)

    def _palace_path(self, conversation_id: str) -> str:
        """This conversation's palace — its own store, not a filtered view."""
        return str(self._conversation_root(conversation_id) / "palace")

    def _wing(self, conversation_id: str) -> str:
        """The wing every drawer of this conversation is filed under."""
        return f"conv-{self._slug(conversation_id)}"

    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Write the session to disk verbatim, then mine it into the palace.

        The file name carries the session id, which is what a hit's
        ``source_file`` gives back; the number keeps the directory in
        chronological order for anyone reading it.
        """
        if not session.turns:
            return
        self._conversation_id = conversation_id
        # one turn per line, and the line is what a hit is matched against;
        # each is stripped so a drawer's own stripping cannot change it
        lines = [
            (f"{turn.speaker}: {turn.text}".strip(), turn.turn_id)
            for turn in session.turns
        ]
        body = "\n".join(line for line, _ in lines).strip()
        sessions_dir = self._conversation_root(conversation_id) / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = (
            sessions_dir
            / f"{len(self._staged):04d}-{self._slug(session.session_id)}.txt"
        )
        path.write_text(body, encoding="utf-8")
        self._staged[path.name] = _StagedSession(
            session_id=session.session_id, lines=lines
        )
        self._mine(conversation_id, path)

    def _mine(self, conversation_id: str, path: Path) -> None:
        """Mine one session file into this conversation's palace.

        `files=[path]` hands MemPalace the file directly, so a conversation
        of N sessions does not re-walk the whole staging directory N times.
        """
        from mempalace.miner import mine

        # mine narrates its progress to stdout, which the harness owns;
        # failures raise, so nothing diagnostic is lost by dropping it
        with _mute().muted():
            mine(
                str(path.parent),
                self._palace_path(conversation_id),
                wing_override=self._wing(conversation_id),
                agent="amb",
                respect_gitignore=False,
                files=[path],
            )

    def remember(
        self,
        conversation_id: str,
        content: str,
        *,
        session_id: str,
        turn_ids: list[str],
        room: str = DEFAULT_ROOM,
        chunk_index: int | None = None,
    ) -> str:
        """File one drawer directly, without staging a file for it.

        The write path behind the agentic toolset: an agent storing a fact
        live has no source file, so the drawer is filed under a synthetic
        one and its provenance is kept here, keyed by the drawer id the
        recipe derives deterministically from what it is filed as.
        """
        from mempalace.ids import make_drawer_id_from_chunk
        from mempalace.miner import add_drawer
        from mempalace.palace import get_collection

        self._conversation_id = conversation_id
        wing = self._wing(conversation_id)
        room = self._slug(room) if room else DEFAULT_ROOM
        source_file = AGENT_SOURCE.format(session_id=self._slug(session_id))
        index = len(self._written) if chunk_index is None else chunk_index
        drawer_id = make_drawer_id_from_chunk(wing, room, source_file, index)
        collection = get_collection(self._palace_path(conversation_id))
        add_drawer(collection, wing, room, content, source_file, index, "amb")
        self._written[drawer_id] = (session_id, list(turn_ids))
        return drawer_id

    def recall_hits(
        self,
        conversation_id: str,
        *,
        query: str,
        room: str | None = None,
        k: int = 10,
    ) -> list[MemoryHit]:
        """Recall inside the conversation, with provenance restored.

        The question travels verbatim: MemPalace tokenises it itself for the
        BM25 half of the re-rank and embeds it whole for the vector half.
        `room` narrows the search to one topic of the palace — the agentic
        toolset's to pass, since a room is the agent's own filing choice.

        Raises:
            RuntimeError: if MemPalace could not search the palace.
        """
        from mempalace.searcher import search_memories

        result = search_memories(
            query=query,
            palace_path=self._palace_path(conversation_id),
            wing=self._wing(conversation_id),
            room=self._slug(room) if room else None,
            n_results=k,
            candidate_strategy=self.candidate_strategy,
        )
        # search_memories reports failure as an envelope rather than raising,
        # and an unraised failure would score as "retrieved nothing" — a
        # silent zero. Raise so the run records it as the error it is.
        if error := result.get("error"):
            logger.bind(scope="mempalace").error(
                "search failed: conversation={!r} query={!r} error={!r} details={!r}",
                conversation_id,
                query,
                error,
                result.get("details"),
            )
            raise RuntimeError(f"mempalace search failed: {error}")
        hits = []
        for hit in result.get("results") or []:
            text = hit.get("text") or ""
            drawer_id = str(hit.get("drawer_id") or "")
            session_id, turn_ids = self._provenance(hit, text, drawer_id)
            hits.append(
                MemoryHit(
                    content=text,
                    score=hit.get("similarity"),
                    turn_ids=turn_ids,
                    session_ids=[session_id] if session_id else [],
                    metadata={
                        "drawer_id": drawer_id,
                        "room": hit.get("room"),
                        "matched_via": hit.get("matched_via"),
                    },
                )
            )
        return hits[:k]

    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k drawers for the query, across the whole palace."""
        if not query.strip():
            return []
        return self.recall_hits(conversation_id, query=query, k=k)

    def _provenance(
        self, hit: dict, text: str, drawer_id: str
    ) -> tuple[str | None, list[str]]:
        """What this drawer attests: its session, and the turns it holds whole."""
        if written := self._written.get(drawer_id):
            return written
        staged = self._staged.get(str(hit.get("source_file") or ""))
        if staged is None:
            return None, []
        return staged.session_id, staged.turns_within(text)

    def _drawer_count(self, conversation_id: str) -> int | None:
        """How many drawers this conversation's palace holds, if it can say."""
        try:
            from mempalace.palace import get_collection

            return int(get_collection(self._palace_path(conversation_id)).count())
        except Exception:  # noqa: BLE001 - a footprint number, never a failure
            return None

    def teardown(self) -> None:
        """Delete this conversation's palace and its staged session files.

        Scoped to one directory on purpose: the palaces of the conversations
        running beside this one under `--workers N` are siblings, and a
        wider sweep would take them with it.
        """
        if self._conversation_id is not None:
            shutil.rmtree(
                self._conversation_root(self._conversation_id), ignore_errors=True
            )
        self._staged.clear()
        self._written.clear()

    def stats(self) -> dict:
        """Report what this run stored, and how it retrieved."""
        stats: dict = {
            "sessions": len(self._staged),
            "agent_drawers": len(self._written),
            "candidate_strategy": self.candidate_strategy,
        }
        if self._conversation_id and (
            count := self._drawer_count(self._conversation_id)
        ):
            stats["drawers"] = count
        if self.embedding_model:
            stats["embedding_model"] = self.embedding_model
            stats["embedder_is_local"] = self.embedding_model in LOCAL_EMBEDDERS
        return stats
