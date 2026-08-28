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

"""EverOS's tool surface for `--mode agentic`.

The agent drives EverOS's own verbs: `remember` hands it a message and
its pipeline extracts from it, `recall` searches what it extracted. The
project and app stay the adapter's, never the agent's to choose, so
conversation isolation cannot be argued away by a tool call.

`remember` closes the session boundary on every call, because an agent
told "stored" about messages still sitting accumulated would go on to
search for them and find nothing.
"""

import time
from typing import Any

from amb.agent.toolset import IngestToolset, SearchToolset
from amb.base import Memory
from amb.contracts import Session
from src.memory import EverOSMemory


class EverOSSearchToolset(SearchToolset):
    """EverOS's search, exposed to the answering agent."""

    def __init__(
        self, memory: Memory, conversation_id: str, k: int = 10, **kwargs: Any
    ) -> None:
        """Bind the toolset to one conversation and expose recall."""
        super().__init__(memory, conversation_id, k=k, **kwargs)
        self.add_function(self.recall, name="recall")

    @property
    def everos(self) -> EverOSMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, EverOSMemory)
        return self.memory

    def recall(self, query: str) -> list[dict]:
        """Search the memories extracted from this conversation.

        Args:
            query: The information need as a natural phrase; matched
                against the episodes and the facts drawn from them.

        Returns:
            The best-matching remembered episodes.
        """
        t0 = time.perf_counter()
        hits = self.everos.search(self.conversation_id, query, k=self.k)
        return self.record(hits, time.perf_counter() - t0)


class EverOSIngestToolset(IngestToolset):
    """EverOS's write path, exposed to the ingesting agent."""

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
    def everos(self) -> EverOSMemory:
        """The bound system, typed to its concrete class."""
        assert isinstance(self.memory, EverOSMemory)
        return self.memory

    def remember(self, content: str, source_turn_ids: list[str]) -> str:
        """Hand one message to EverOS to remember.

        Args:
            content: What is worth remembering, phrased so EverOS's own
                extraction has something to work with — a statement, not
                a keyword.
            source_turn_ids: The ids of the turns it came from, exactly
                as shown in the transcript.

        Returns:
            Whether the content was stored, and whether it was extracted.
        """
        cited = [t for t in source_turn_ids if t in self.turn_ids()]
        if not cited:
            return "not stored: none of the cited turn ids exist in this session"
        t0 = time.perf_counter()
        status = self.everos.memorize(
            self.conversation_id,
            [
                {
                    "sender_id": self.everos._user_id(self.conversation_id),
                    "sender_name": "agent",
                    "role": "assistant",
                    "timestamp": self.everos._epoch_ms(self.session.timestamp),
                    "content": content,
                }
            ],
            session_id=self.session.session_id,
        )
        self.record_write(time.perf_counter() - t0)
        if len(cited) < len(source_turn_ids):
            return f"stored ({status}), but unknown turn ids were dropped"
        return f"stored ({status})"
