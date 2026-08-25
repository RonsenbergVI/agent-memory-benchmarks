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

"""MemPalace's tool surface for `--mode agentic`.

The agent drives MemPalace's own filing: `remember` puts one verbatim
passage in a *room* of the conversation's palace, `recall` searches that
palace and can narrow to a room it filed into earlier. Rooms are the
system's structured index, and choosing one is exactly the decision
MemPalace asks a writer to make — so it is the agent's, not the adapter's.

Which palace is not: the wing and the palace directory stay the adapter's,
so conversation isolation cannot be argued away by a tool call.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import DEFAULT_ROOM, MemPalaceMemory


class MemPalaceSearchToolset(SearchToolset):
    """MemPalace's recall, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose recall."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.recall, name="recall")

    @property
    def mempalace(self) -> MemPalaceMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, MemPalaceMemory)
        return self.memory

    def recall(self, query: str, room: str | None = None) -> list[dict]:
        """Search the palace for what was remembered earlier.

        Args:
            query: The information need as a natural phrase; matched against
                the stored text both semantically and by keyword.
            room: Optional topic to search inside, as named when the passage
                was remembered. Omit it to search the whole palace.

        Returns:
            The best-matching remembered passages, verbatim.
        """
        t0 = time.perf_counter()
        hits = self.mempalace.recall_hits(
            self.conversation_id, query=query, room=room, k=self.k
        )
        return self.record(hits, time.perf_counter() - t0)


class MemPalaceIngestToolset(IngestToolset):
    """MemPalace's write path, exposed to the ingesting agent."""

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
    def mempalace(self) -> MemPalaceMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, MemPalaceMemory)
        return self.memory

    def remember(
        self, passage: str, source_turn_ids: list[str], room: str = DEFAULT_ROOM
    ) -> str:
        """File one passage in the palace, verbatim.

        Args:
            passage: The text worth keeping, in the words it was said in —
                MemPalace stores what it is given and never paraphrases, so
                a summary here is stored as a summary.
            source_turn_ids: The ids of the turns the passage came from,
                exactly as shown in the transcript.
            room: The topic to file it under, so a later recall can be
                scoped to it. Reuse a room name across related passages.

        Returns:
            Whether the passage was filed.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not filed: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.mempalace.remember(
            self.conversation_id,
            passage,
            session_id=self.session.session_id,
            turn_ids=cited,
            room=room,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "filed, but unknown turn ids were dropped from the citation"
        return "filed"
