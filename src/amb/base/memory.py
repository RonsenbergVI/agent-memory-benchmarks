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
from typing import ClassVar

from amb.contracts.conversation import Session
from amb.contracts.memory import MemoryHit


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

    def __init__(self, **params: object) -> None:
        """Keep any `--param` overrides the CLI passed through."""
        self.params = params

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

    def teardown(self) -> None:
        """Release resources / delete stored memories for this sample."""

    def stats(self) -> dict:
        """Optional footprint numbers (stored items, bytes, ...)."""
        return {}

    def usage_counters(self) -> dict:
        """Token spend this system computes about itself (empty by default).

        For spend inside the system's own server, invisible to the SDK
        tracker: report OpenAIUsageTracker.KEYS here (letta: tiktoken over
        every stored passage and query); TiktokenUsageTracker books the
        deltas into the report.
        """
        return {}
