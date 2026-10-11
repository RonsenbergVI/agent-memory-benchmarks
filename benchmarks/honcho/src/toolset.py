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

"""Honcho's tool surface for `--mode agentic`.

The agent drives Honcho's own verbs, named as its MCP server names them:
`add_messages_to_session` writes a message from a speaker, which the
deriver then reasons over, and `search` runs the workspace's message
search. The workspace stays the adapter's, never the agent's to choose,
so conversation isolation cannot be argued away by a tool call.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import HonchoMemory


class HonchoSearchToolset(SearchToolset):
    """Honcho's message search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose search."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.search, name="search")

    @property
    def honcho(self) -> HonchoMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, HonchoMemory)
        return self.memory

    def search(self, query: str) -> list[dict]:
        """Search this conversation's messages.

        Args:
            query: The information need as a natural phrase; matched both
                semantically and by full text.

        Returns:
            The best-matching messages, each prefixed with its speaker.
        """
        t0 = time.perf_counter()
        hits = self.honcho.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class HonchoIngestToolset(IngestToolset):
    """Honcho's message write path, exposed to the ingesting agent."""

    def __init__(
        self,
        memory: Memory,
        conversation_id: str,
        session: Session,
        **kwargs: Any,
    ) -> None:
        """Bind the toolset to one session and expose add_messages_to_session."""
        super().__init__(memory, conversation_id, session, **kwargs)
        self.add_function(self.add_messages_to_session, name="add_messages_to_session")

    @property
    def honcho(self) -> HonchoMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, HonchoMemory)
        return self.memory

    def add_messages_to_session(self, content: str, source_turn_ids: list[str]) -> str:
        """Record one message worth remembering, as said by the cited speaker.

        Args:
            content: What is worth remembering, phrased so it reads
                correctly out of context.
            source_turn_ids: The ids of the turns it came from, exactly as
                shown in the transcript; the first one's speaker is who
                the message is attributed to.

        Returns:
            Whether the message was recorded.
        """
        speakers = {turn.turn_id: turn.speaker for turn in self.session.turns}
        cited = [t for t in source_turn_ids if t in speakers]
        if not cited:
            return "not recorded: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        self.honcho.store(
            self.conversation_id,
            self.session,
            [(speakers[cited[0]], content, cited)],
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return "recorded, but unknown turn ids were dropped from the citation"
        return "recorded"
