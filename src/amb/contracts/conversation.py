# MIT License

# Copyright (c) 2026 René-Jean Corneille

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """A single utterance within a session."""

    turn_id: str
    speaker: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Session(BaseModel):
    """A contiguous block of turns.

    Sessions are typically separated from each other by hours or days of
    in-world time.
    """

    session_id: str
    timestamp: str | None = None
    turns: list[Turn] = Field(default_factory=list)

    def __str__(self) -> str:
        """Render the session as plain text for systems that ingest prose."""
        header = f"[session {self.session_id}"
        if self.timestamp:
            header += f" @ {self.timestamp}"
        header += "]"
        lines = [header] + [f"{t.speaker}: {t.text}" for t in self.turns]
        return "\n".join(lines)


class Conversation(BaseModel):
    """A full multi-session history between fixed participants."""

    conversation_id: str
    speakers: list[str] = Field(default_factory=list)
    sessions: list[Session] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        """Total turns across every session."""
        return sum(len(s.turns) for s in self.sessions)


class QAPair(BaseModel):
    """A ground-truth probing question about a conversation."""

    question_id: str
    question: str
    answer: str | None = None
    category: str | None = None
    # ids of turns and/or sessions that contain the evidence for the answer
    evidence_turn_ids: list[str] = Field(default_factory=list)
    evidence_session_ids: list[str] = Field(default_factory=list)
    # in-world date at which the question is asked (LongMemEval)
    question_date: str | None = None
    metadata: dict = Field(default_factory=dict)


class Sample(BaseModel):
    """One evaluation unit: a conversation plus its probing questions."""

    sample_id: str
    dataset: str
    conversation: Conversation
    qa: list[QAPair] = Field(default_factory=list)
