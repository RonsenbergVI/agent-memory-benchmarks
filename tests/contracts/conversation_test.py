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

import pytest
from pydantic import ValidationError

from amb.contracts.conversation import Conversation, QAPair, Sample, Session, Turn


def test_turn_required_fields_are_stored_verbatim():
    turn = Turn(turn_id="t1", speaker="user", text="hello")
    assert turn.turn_id == "t1"
    assert turn.speaker == "user"
    assert turn.text == "hello"


def test_turn_metadata_defaults_to_empty_dict():
    turn = Turn(turn_id="t1", speaker="user", text="hello")
    assert turn.metadata == {}


def test_turn_metadata_defaults_are_independent_instances():
    first = Turn(turn_id="t1", speaker="user", text="a")
    second = Turn(turn_id="t2", speaker="user", text="b")
    first.metadata["seen"] = True
    assert second.metadata == {}


@pytest.mark.parametrize("missing", ["turn_id", "speaker", "text"])
def test_turn_missing_required_field_is_rejected(missing):
    payload = {"turn_id": "t1", "speaker": "user", "text": "hello"}
    del payload[missing]
    with pytest.raises(ValidationError) as exc_info:
        Turn.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == (missing,)


def test_integer_turn_id_is_not_coerced_to_str():
    payload = {"turn_id": 1, "speaker": "user", "text": "hello"}
    with pytest.raises(ValidationError) as exc_info:
        Turn.model_validate(payload)
    assert exc_info.value.errors()[0]["loc"] == ("turn_id",)
    assert exc_info.value.errors()[0]["type"] == "string_type"


def test_turn_non_dict_metadata_is_rejected():
    payload = {
        "turn_id": "t1",
        "speaker": "user",
        "text": "hello",
        "metadata": ["not", "a", "dict"],
    }
    with pytest.raises(ValidationError) as exc_info:
        Turn.model_validate(payload)
    assert exc_info.value.errors()[0]["loc"] == ("metadata",)
    assert exc_info.value.errors()[0]["type"] == "dict_type"


def test_turn_unknown_keys_are_ignored():
    payload = {
        "turn_id": "t1",
        "speaker": "user",
        "text": "hello",
        "unexpected": "extra",
    }
    turn = Turn.model_validate(payload)
    assert "unexpected" not in turn.model_dump()


def test_turn_model_dump_shape():
    turn = Turn(turn_id="t1", speaker="user", text="hello")
    assert turn.model_dump() == {
        "turn_id": "t1",
        "speaker": "user",
        "text": "hello",
        "metadata": {},
    }


def test_only_session_id_is_required():
    session = Session(session_id="s1")
    assert session.session_id == "s1"
    assert session.timestamp is None
    assert session.turns == []


def test_missing_session_id_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        Session.model_validate({})
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("session_id",)


def test_session_nested_turn_dicts_are_promoted_to_models():
    session = Session.model_validate(
        {
            "session_id": "s1",
            "timestamp": "2024-01-01",
            "turns": [{"turn_id": "t1", "speaker": "user", "text": "hi"}],
        }
    )
    assert session.turns == [Turn(turn_id="t1", speaker="user", text="hi")]


def test_session_invalid_nested_turn_reports_nested_location():
    payload = {"session_id": "s1", "turns": [{"turn_id": "t1", "text": "hi"}]}
    with pytest.raises(ValidationError) as exc_info:
        Session.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("turns", 0, "speaker")


def test_session_non_list_turns_is_rejected():
    payload = {"session_id": "s1", "turns": "not a list"}
    with pytest.raises(ValidationError) as exc_info:
        Session.model_validate(payload)
    assert exc_info.value.errors()[0]["loc"] == ("turns",)
    assert exc_info.value.errors()[0]["type"] == "list_type"


def test_session_non_string_timestamp_is_rejected():
    payload = {"session_id": "s1", "timestamp": 1704067200}
    with pytest.raises(ValidationError) as exc_info:
        Session.model_validate(payload)
    assert exc_info.value.errors()[0]["loc"] == ("timestamp",)


def test_only_conversation_id_is_required():
    conversation = Conversation(conversation_id="c1")
    assert conversation.conversation_id == "c1"
    assert conversation.speakers == []
    assert conversation.sessions == []
    assert conversation.metadata == {}


def test_missing_conversation_id_is_rejected():
    with pytest.raises(ValidationError) as exc_info:
        Conversation.model_validate({})
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("conversation_id",)


def test_conversation_list_defaults_are_independent_instances():
    first = Conversation(conversation_id="c1")
    second = Conversation(conversation_id="c2")
    first.speakers.append("Ana")
    first.sessions.append(Session(session_id="s1"))
    assert second.speakers == []
    assert second.sessions == []


def test_conversation_non_string_speaker_is_rejected():
    payload = {"conversation_id": "c1", "speakers": ["Ana", 2]}
    with pytest.raises(ValidationError) as exc_info:
        Conversation.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "string_type"
    assert errors[0]["loc"] == ("speakers", 1)


def test_only_question_id_and_question_are_required():
    qa = QAPair(question_id="q1", question="Who spoke first?")
    assert qa.question_id == "q1"
    assert qa.question == "Who spoke first?"
    assert qa.answer is None
    assert qa.category is None
    assert qa.evidence_turn_ids == []
    assert qa.evidence_session_ids == []
    assert qa.question_date is None
    assert qa.metadata == {}


@pytest.mark.parametrize("missing", ["question_id", "question"])
def test_qa_pair_missing_required_field_is_rejected(missing):
    payload = {"question_id": "q1", "question": "Who spoke first?"}
    del payload[missing]
    with pytest.raises(ValidationError) as exc_info:
        QAPair.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == (missing,)


def test_qa_pair_bare_string_evidence_is_not_promoted_to_list():
    payload = {
        "question_id": "q1",
        "question": "Who spoke first?",
        "evidence_turn_ids": "t1",
    }
    with pytest.raises(ValidationError) as exc_info:
        QAPair.model_validate(payload)
    assert exc_info.value.errors()[0]["loc"] == ("evidence_turn_ids",)
    assert exc_info.value.errors()[0]["type"] == "list_type"


def test_qa_pair_explicit_none_answer_is_accepted():
    qa = QAPair.model_validate(
        {"question_id": "q1", "question": "Who spoke first?", "answer": None}
    )
    assert qa.answer is None


def test_sample_qa_defaults_to_empty_list():
    sample = Sample(
        sample_id="locomo-0",
        dataset="locomo",
        conversation=Conversation(conversation_id="c1"),
    )
    assert sample.qa == []


@pytest.mark.parametrize("missing", ["sample_id", "dataset", "conversation"])
def test_sample_missing_required_field_is_rejected(missing):
    payload = {
        "sample_id": "locomo-0",
        "dataset": "locomo",
        "conversation": {"conversation_id": "c1"},
    }
    del payload[missing]
    with pytest.raises(ValidationError) as exc_info:
        Sample.model_validate(payload)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == (missing,)


def test_sample_nested_payload_is_promoted_to_models():
    sample = Sample.model_validate(
        {
            "sample_id": "locomo-0",
            "dataset": "locomo",
            "conversation": {
                "conversation_id": "c1",
                "sessions": [
                    {
                        "session_id": "s1",
                        "turns": [{"turn_id": "t1", "speaker": "Ana", "text": "hi"}],
                    }
                ],
            },
            "qa": [{"question_id": "q1", "question": "Who spoke first?"}],
        }
    )
    assert isinstance(sample.conversation, Conversation)
    assert isinstance(sample.conversation.sessions[0], Session)
    assert isinstance(sample.conversation.sessions[0].turns[0], Turn)
    assert isinstance(sample.qa[0], QAPair)


def _full_sample() -> Sample:
    return Sample(
        sample_id="locomo-0",
        dataset="locomo",
        conversation=Conversation(
            conversation_id="c1",
            speakers=["Ana", "Ben"],
            sessions=[
                Session(
                    session_id="s1",
                    timestamp="2024-01-01",
                    turns=[
                        Turn(
                            turn_id="t1",
                            speaker="Ana",
                            text="hi",
                            metadata={"lang": "en"},
                        )
                    ],
                )
            ],
            metadata={"source": "unit-test"},
        ),
        qa=[
            QAPair(
                question_id="q1",
                question="Who greeted first?",
                answer="Ana",
                category="single-hop",
                evidence_turn_ids=["t1"],
                evidence_session_ids=["s1"],
                question_date="2024-02-01",
                metadata={"difficulty": "easy"},
            )
        ],
    )


def test_full_sample_model_dump_shape():
    assert _full_sample().model_dump() == {
        "sample_id": "locomo-0",
        "dataset": "locomo",
        "conversation": {
            "conversation_id": "c1",
            "speakers": ["Ana", "Ben"],
            "sessions": [
                {
                    "session_id": "s1",
                    "timestamp": "2024-01-01",
                    "turns": [
                        {
                            "turn_id": "t1",
                            "speaker": "Ana",
                            "text": "hi",
                            "metadata": {"lang": "en"},
                        }
                    ],
                }
            ],
            "metadata": {"source": "unit-test"},
        },
        "qa": [
            {
                "question_id": "q1",
                "question": "Who greeted first?",
                "answer": "Ana",
                "category": "single-hop",
                "evidence_turn_ids": ["t1"],
                "evidence_session_ids": ["s1"],
                "question_date": "2024-02-01",
                "metadata": {"difficulty": "easy"},
            }
        ],
    }


def test_full_sample_round_trip_is_lossless():
    original = _full_sample()
    assert Sample.model_validate(original.model_dump()) == original


def test_minimal_sample_round_trip_preserves_defaults():
    original = Sample(
        sample_id="locomo-1",
        dataset="locomo",
        conversation=Conversation(conversation_id="c2"),
    )
    assert Sample.model_validate(original.model_dump()) == original


def test_qapair_round_trip_preserves_none_fields():
    original = QAPair(question_id="q1", question="Who spoke first?")
    restored = QAPair.model_validate(original.model_dump())
    assert restored == original
    assert restored.answer is None
    assert restored.question_date is None
