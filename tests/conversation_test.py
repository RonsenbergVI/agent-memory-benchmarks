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

from amb.contracts.conversation import Conversation, Session, Turn


def _turn(turn_id: str, speaker: str, text: str) -> Turn:
    return Turn(turn_id=turn_id, speaker=speaker, text=text)


def test_session_str_header_only_when_no_turns():
    session = Session(session_id="s1")
    assert str(session) == "[session s1]"


def test_session_str_header_includes_timestamp_when_present():
    session = Session(session_id="s1", timestamp="2026-01-02 09:30")
    assert str(session) == "[session s1 @ 2026-01-02 09:30]"


def test_session_str_empty_timestamp_is_treated_as_absent():
    # An empty string is falsy, so the "@ ..." segment is omitted.
    session = Session(session_id="s1", timestamp="")
    assert str(session) == "[session s1]"


def test_session_str_turns_render_as_speaker_colon_text_lines():
    session = Session(
        session_id="s2",
        timestamp="2026-03-04",
        turns=[
            _turn("t1", "alice", "hi bob"),
            _turn("t2", "bob", "hi alice"),
        ],
    )
    expected = "[session s2 @ 2026-03-04]\nalice: hi bob\nbob: hi alice"
    assert str(session) == expected


def test_session_str_turn_order_is_preserved_in_rendering():
    turns = [_turn(f"t{i}", "spk", f"line {i}") for i in range(5)]
    session = Session(session_id="s3", turns=turns)
    lines = str(session).splitlines()
    assert lines[1:] == [f"spk: line {i}" for i in range(5)]


def test_session_str_does_not_escape_turn_text_newlines():
    # A newline inside a turn's text flows straight into the output,
    # producing an extra physical line.
    session = Session(session_id="s4", turns=[_turn("t1", "alice", "a\nb")])
    assert str(session) == "[session s4]\nalice: a\nb"


def test_session_str_has_no_trailing_newline():
    session = Session(session_id="s5", turns=[_turn("t1", "alice", "hello")])
    assert not str(session).endswith("\n")


def test_num_turns_zero_when_no_sessions():
    conversation = Conversation(conversation_id="c1")
    assert conversation.num_turns == 0


def test_num_turns_zero_when_sessions_have_no_turns():
    conversation = Conversation(
        conversation_id="c1",
        sessions=[Session(session_id="s1"), Session(session_id="s2")],
    )
    assert conversation.num_turns == 0


def test_num_turns_sums_turns_across_all_sessions():
    conversation = Conversation(
        conversation_id="c1",
        speakers=["alice", "bob"],
        sessions=[
            Session(
                session_id="s1",
                turns=[_turn("t1", "alice", "a"), _turn("t2", "bob", "b")],
            ),
            Session(session_id="s2", turns=[]),
            Session(
                session_id="s3",
                turns=[
                    _turn("t3", "alice", "c"),
                    _turn("t4", "bob", "d"),
                    _turn("t5", "alice", "e"),
                ],
            ),
        ],
    )
    assert conversation.num_turns == 5


def test_num_turns_is_computed_not_cached():
    session = Session(session_id="s1", turns=[_turn("t1", "alice", "a")])
    conversation = Conversation(conversation_id="c1", sessions=[session])
    assert conversation.num_turns == 1
    session.turns.append(_turn("t2", "bob", "b"))
    assert conversation.num_turns == 2


def test_num_turns_counts_duplicate_turn_ids_individually():
    # num_turns counts list entries; it does not deduplicate by turn_id.
    conversation = Conversation(
        conversation_id="c1",
        sessions=[
            Session(
                session_id="s1",
                turns=[_turn("t1", "alice", "a"), _turn("t1", "alice", "a")],
            ),
        ],
    )
    assert conversation.num_turns == 2
