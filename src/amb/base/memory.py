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

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from amb.contracts.conversation import Session
from amb.contracts.memory import MemoryHit

# what --llm / --embedding configure; nothing else may set them
MODEL_PARAMS = ("model", "embedding_model")


def split_model(spec: str) -> tuple[str, str]:
    """Split a `provider:name` model id into its two halves.

    Raises:
        ValueError: when the id carries no provider.
    """
    provider, sep, name = spec.partition(":")
    if not sep or not provider or not name:
        raise ValueError(
            f"model id {spec!r} must be provider:name, e.g. openai:gpt-5-mini"
        )
    return provider, name


def spanned_turns(session: Session, text: str) -> list[str]:
    """Ids of the session's turns that a verbatim slice of its transcript covers.

    The transcript is the `speaker: text` lines joined by newlines, as the
    chunking systems ingest it. Returns [] when `text` is not a slice of it.
    """
    lines = [f"{turn.speaker}: {turn.text}" for turn in session.turns]
    needle = text.strip()
    start = "\n".join(lines).find(needle) if needle else -1
    if start < 0:
        return []
    end, offset, spanned = start + len(needle), 0, []
    for turn, line in zip(session.turns, lines, strict=True):
        if offset < end and start < offset + len(line):
            spanned.append(turn.turn_id)
        offset += len(line) + 1
    return spanned


class Memory(ABC):
    """Contract every memory system integration implements.

    The benchmark creates one instance per sample:
    setup -> ingest_session (per session, in order) -> search (per question)
    -> teardown.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    # PyPI distribution whose version identifies the system under test
    sdk_dist: ClassVar[str | None] = None
    # Token-spend accounting: "full" (the only comparable cost column),
    # "partial" (real but short of the truth), "none" (invisible spend whose
    # zero would read as "free"). Reporting renders the three differently.
    usage_coverage: ClassVar[str] = "full"
    # providers the system's own LLM / embedder runs on natively; empty: none
    llm_providers: ClassVar[tuple[str, ...]] = ()
    embedding_providers: ClassVar[tuple[str, ...]] = ()
    # the halves (`llm`, `embedding`) the system can run without, via `none`
    optional_models: ClassVar[tuple[str, ...]] = ()
    # the finest source unit the system itself keeps for a memory, which
    # resolve_turns attributes a hit to: turn, window, segment, chunk, session
    turn_provenance: ClassVar[str] = "session"

    def __init__(self, **params: object) -> None:
        """Keep any `--param` overrides the CLI passed through.

        Raises:
            ValueError: when --llm/--embedding reach a system that consumes
                neither, rather than being silently ignored.
        """
        unused = [key for key in MODEL_PARAMS if key in params]
        if unused:
            raise ValueError(f"{self.name} calls no LLM or embedder it could set")
        self.params = params

    @classmethod
    def model_name(
        cls, spec: str | None, kind: Literal["llm", "embedding"]
    ) -> str | None:
        """The SDK-facing name in a `provider:name` id this system can run.

        Raises:
            ValueError: when the system cannot run that provider natively —
                a model is never swapped for another or shimmed — or cannot
                run without one (`none`).
        """
        if spec is None:
            if kind not in cls.optional_models:
                raise ValueError(f"{cls.name} cannot run without an {kind}")
            return None
        provider, name = split_model(spec)
        supported = cls.llm_providers if kind == "llm" else cls.embedding_providers
        if provider not in supported:
            raise ValueError(
                f"{cls.name} cannot run its {kind} on {provider!r} natively "
                f"(supports: {', '.join(supported) or 'none'})"
            )
        return name

    def coverage(self) -> str:
        """How much of this instance's spend the run accounts for (`usage_coverage`)."""
        return self.usage_coverage

    def models(self) -> dict[str, str | None]:
        """The models this system calls internally; recorded in every run.

        None means the system uses none, or its SDK's own default. Reads
        the conventional `model` / `embedding_model` attributes, so
        adapters rarely need to override this.
        """
        return {
            "ingestion_model": getattr(self, "model", None),
            "embedding_model": getattr(self, "embedding_model", None),
        }

    def version(self) -> str | None:
        """Version of the system being measured; recorded in every run.

        Defaults to `sdk_dist`'s installed version. Server-backed systems
        override this to report the server's version instead.
        """
        if self.sdk_dist is None:
            return None
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(self.sdk_dist)
        except PackageNotFoundError:
            return None

    def setup(self) -> None:
        """Connect to backing services / initialize state."""

    @abstractmethod
    def ingest_session(self, conversation_id: str, session: Session) -> None:
        """Store one conversation session. Called in chronological order."""

    @abstractmethod
    def search(self, conversation_id: str, query: str, k: int = 10) -> list[MemoryHit]:
        """Return up to k pieces of remembered context relevant to query."""

    def resolve_turns(
        self, hits: list[MemoryHit], sessions: list[Session]
    ) -> list[MemoryHit]:
        """The hits with `turn_ids` filled from the provenance search returned.

        Called once the agent has answered: outside the timed search, and
        never seen by the agent. Overrides narrow a hit to the source unit
        the system keeps (`turn_provenance`); a hit left without turn ids
        is attributed to every ingested turn of its sessions.
        """
        turns = {s.session_id: [t.turn_id for t in s.turns] for s in sessions}
        return [
            hit
            if hit.turn_ids
            else hit.model_copy(
                update={
                    "turn_ids": [
                        turn_id
                        for session_id in dict.fromkeys(hit.session_ids)
                        for turn_id in turns.get(session_id, [])
                    ]
                }
            )
            for hit in hits
        ]

    def teardown(self) -> None:
        """Release resources / delete stored memories for this sample."""

    def stats(self) -> dict:
        """Optional footprint numbers (stored items, bytes, ...)."""
        return {}

    def usage_counters(self) -> dict:
        """Token spend this system computes about itself (empty by default).

        For spend inside the system's own server, invisible to the SDK
        tracker: report OpenAIUsageTracker.KEYS here (hindsight: the usage
        its server reports); TiktokenUsageTracker books the deltas into
        the report.
        """
        return {}
