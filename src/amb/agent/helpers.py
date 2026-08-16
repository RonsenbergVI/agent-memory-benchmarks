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

from amb.contracts import MemoryHit, Session


def format_context(hits: list[MemoryHit]) -> str:
    """Render retrieved memories as numbered context for the prompt."""
    if not hits:
        return "(the memory system returned nothing)"
    return "\n\n".join(f"[memory {i + 1}]\n{h.content}" for i, h in enumerate(hits))


def format_session(session: Session) -> str:
    """Render one session as the transcript the ingest agent reads.

    What a live agent would have seen — speakers, text, the date — plus a
    turn marker per line, the citation handle write tools take for
    provenance. Conversation and session ids stay harness-side: an agent
    storing memories in real time knows neither.
    """
    header = (
        f"Conversation of {session.timestamp}" if session.timestamp else "Conversation"
    )
    lines = [f"{turn.turn_id} | {turn.speaker}: {turn.text}" for turn in session.turns]
    return header + "\n" + "\n".join(lines)
