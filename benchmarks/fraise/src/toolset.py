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

"""Fraise's native tool surface for `--mode agentic`.

The agent drives fraise's own verbs: `remember` (with topics and entities of
its choosing) at ingestion, `recall` (keywords, semantic query, topic and
entity filters) at search. Conversation isolation stays forced — the
`conv-<id>` anchor is added by the adapter, never left to the agent.
"""

import re
import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import FraiseMemory


def _slug(topic: str) -> str:
    """Reduce an agent-chosen topic to a safe FQL anchor token."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", topic).strip("-").lower()


class FraiseSearchToolset(SearchToolset):
    """Fraise's recall, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose recall."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.recall, name="recall")

    @property
    def fraise(self) -> FraiseMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, FraiseMemory)
        return self.memory

    def recall(
        self,
        keywords: list[str],
        query: str | None = None,
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> list[dict]:
        """Search the conversation's memory database.

        Args:
            keywords: Words to match in stored facts (full-text seeds).
            query: The information need as a natural phrase; used for
                semantic matching when the database has an embedder.
            topics: Topic tags to require on results — a hard filter, so
                only use tags likely used at storage time.
            entities: People or things results must be about (e.g. a
                speaker's name).

        Returns:
            The best-matching stored facts with their scores.
        """
        t0 = time.perf_counter()
        hits = self.fraise.recall_hits(
            self.conversation_id,
            keywords=keywords,
            query=query,
            topics=[_slug(t) for t in topics or [] if _slug(t)],
            entities=[e.replace("'", "’") for e in entities or []],
            k=self.k,
        )
        return self.record(hits, time.perf_counter() - t0)


class FraiseIngestToolset(IngestToolset):
    """Fraise's remember, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose remember."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.remember, name="remember")

    @property
    def fraise(self) -> FraiseMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, FraiseMemory)
        return self.memory

    def remember(
        self,
        fact: str,
        source_turn_ids: list[str],
        topics: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> str:
        """Store one fact in the memory database.

        Args:
            fact: One self-contained statement worth remembering, phrased so
                a later search can find it.
            source_turn_ids: The ids of the turns the fact came from, exactly
                as shown in the transcript.
            topics: Topic tags for later filtering (e.g. "travel", "health").
            entities: People or things the fact is about (e.g. speaker names).

        Returns:
            Whether the fact was stored.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.fraise.store(
            self.conversation_id,
            f"{self.fraise.session_header(self.session)} {fact}",
            session_id=self.session.session_id,
            turn_ids=cited,
            topics=[_slug(t) for t in topics or [] if _slug(t)],
            entities=[e.replace("'", "’") for e in entities or []],
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "stored, but unknown turn ids were dropped from the citation"
        return "stored"
