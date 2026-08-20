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

import json
from pathlib import Path

import pytest

from amb.contracts import Sample
from amb.datasets.locomo import LocomoLoader


def record() -> dict:
    """A small handcrafted LoCoMo-shaped conversation with QA annotations."""
    return {
        "sample_id": "conv-9",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I adopted a puppy today!",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "Look at her!",
                    "blip_caption": "a golden retriever puppy on grass",
                },
            ],
            "session_2_date_time": "3:14 pm on 21 May, 2023",
            "session_2": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D2:1",
                    "text": "She has doubled in size already.",
                },
            ],
        },
        "qa": [
            {
                "question": "What pet did Caroline adopt?",
                "answer": "A puppy",
                "evidence": ["D1:1"],
                "category": 4,
            },
            {
                "question": "How many days passed between the two sessions?",
                "answer": 13,
                "evidence": ["D1:1", "D2:1"],
                "category": 2,
            },
            {
                "question": "What did Caroline name her cat?",
                "adv_answer": "Caroline has a puppy, not a cat.",
                "category": 5,
            },
        ],
    }


def make_loader(tmp_path: Path, records: list[dict]) -> LocomoLoader:
    """A loader whose cache is pre-seeded so pull() never touches the network."""
    loader = LocomoLoader(cache_dir=tmp_path)
    (loader.cache_dir / "locomo10.json").write_text(json.dumps(records))
    return loader


def load_single(tmp_path: Path, rec: dict) -> Sample:
    return make_loader(tmp_path, [rec]).load()[0]


def qa_only_record(qa: dict) -> dict:
    return {"sample_id": "s", "conversation": {}, "qa": [qa]}


def test_load_returns_one_sample_per_conversation(tmp_path):
    second = record()
    second["sample_id"] = "conv-10"
    samples = make_loader(tmp_path, [record(), second]).load()
    assert len(samples) == 2
    assert [s.sample_id for s in samples] == ["conv-9", "conv-10"]
    assert [s.dataset for s in samples] == ["locomo", "locomo"]


def test_sample_id_and_conversation_id_come_from_sample_id(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.sample_id == "conv-9"
    assert sample.conversation.conversation_id == "conv-9"


def test_missing_sample_id_falls_back_to_list_index(tmp_path):
    first = record()
    del first["sample_id"]
    second = record()
    del second["sample_id"]
    samples = make_loader(tmp_path, [first, second]).load()
    assert [s.sample_id for s in samples] == ["0", "1"]
    assert samples[1].qa[0].question_id == "1:0"


def test_speakers_come_from_speaker_a_and_speaker_b(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.conversation.speakers == ["Caroline", "Melanie"]


def test_missing_speaker_b_is_dropped(tmp_path):
    rec = record()
    del rec["conversation"]["speaker_b"]
    sample = load_single(tmp_path, rec)
    assert sample.conversation.speakers == ["Caroline"]


def test_sessions_numbered_in_order_with_timestamps(tmp_path):
    sessions = load_single(tmp_path, record()).conversation.sessions
    assert [s.session_id for s in sessions] == ["1", "2"]
    assert sessions[0].timestamp == "1:56 pm on 8 May, 2023"
    assert sessions[1].timestamp == "3:14 pm on 21 May, 2023"


def test_session_without_date_time_has_none_timestamp(tmp_path):
    rec = record()
    del rec["conversation"]["session_2_date_time"]
    sessions = load_single(tmp_path, rec).conversation.sessions
    assert sessions[1].timestamp is None


def test_session_numbering_gap_drops_later_sessions(tmp_path):
    # Current behaviour: collection walks session_1, session_2, ... and stops
    # at the first missing number, silently dropping anything after a gap.
    rec = record()
    conv = rec["conversation"]
    conv["session_3"] = conv.pop("session_2")
    conv["session_3_date_time"] = conv.pop("session_2_date_time")
    sessions = load_single(tmp_path, rec).conversation.sessions
    assert [s.session_id for s in sessions] == ["1"]


def test_dia_id_becomes_turn_id(tmp_path):
    sessions = load_single(tmp_path, record()).conversation.sessions
    assert [t.turn_id for t in sessions[0].turns] == ["D1:1", "D1:2"]
    assert [t.turn_id for t in sessions[1].turns] == ["D2:1"]


def test_turn_speaker_and_text_are_copied_verbatim(tmp_path):
    turn = load_single(tmp_path, record()).conversation.sessions[0].turns[0]
    assert turn.speaker == "Caroline"
    assert turn.text == "I adopted a puppy today!"


def test_blip_caption_is_appended_to_text(tmp_path):
    turn = load_single(tmp_path, record()).conversation.sessions[0].turns[1]
    assert turn.text == "Look at her! [shared image: a golden retriever puppy on grass]"


def test_missing_turn_fields_get_defaults(tmp_path):
    # A bare turn gets a synthetic 0-based id, note the real dia_ids are 1-based.
    rec = record()
    rec["conversation"]["session_1"] = [{}]
    turn = load_single(tmp_path, rec).conversation.sessions[0].turns[0]
    assert turn.turn_id == "D1:0"
    assert turn.speaker == "unknown"
    assert turn.text == ""


def test_caption_only_turn_keeps_leading_space(tmp_path):
    rec = record()
    rec["conversation"]["session_1"] = [{"dia_id": "D1:1", "blip_caption": "sunset"}]
    turn = load_single(tmp_path, rec).conversation.sessions[0].turns[0]
    assert turn.text == " [shared image: sunset]"


def test_question_ids_are_sample_scoped_indices(tmp_path):
    sample = load_single(tmp_path, record())
    assert [q.question_id for q in sample.qa] == ["conv-9:0", "conv-9:1", "conv-9:2"]


@pytest.mark.parametrize(
    ("code", "name"),
    [
        (1, "multi-hop"),
        (2, "temporal"),
        (3, "open-domain"),
        (4, "single-hop"),
        (5, "adversarial"),
    ],
)
def test_category_code_maps_to_taxonomy_name(tmp_path, code, name):
    rec = qa_only_record({"question": "q", "answer": "a", "category": code})
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].category == name


def test_unknown_category_code_is_stringified(tmp_path):
    rec = qa_only_record({"question": "q", "answer": "a", "category": 9})
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].category == "9"


def test_numeric_answer_is_stringified(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.qa[1].answer == "13"


def test_adv_answer_used_when_answer_key_absent(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.qa[2].answer == "Caroline has a puppy, not a cat."


def test_answer_is_none_when_neither_key_present(tmp_path):
    rec = qa_only_record({"question": "q", "category": 3})
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].answer is None


def test_evidence_becomes_turn_ids_stringified(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.qa[1].evidence_turn_ids == ["D1:1", "D2:1"]
    rec = qa_only_record({"question": "q", "answer": "a", "evidence": [5, "D1:2"]})
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].evidence_turn_ids == ["5", "D1:2"]


def test_evidence_session_ids_derived_from_turn_ids(tmp_path):
    sample = load_single(tmp_path, record())
    assert sample.qa[0].evidence_session_ids == ["1"]
    assert sample.qa[1].evidence_session_ids == ["1", "2"]


def test_evidence_session_ids_dedupe_and_sort_lexicographically(tmp_path):
    # Current behaviour: session ids sort as strings, so "10" lands before "2".
    rec = qa_only_record(
        {
            "question": "q",
            "answer": "a",
            "evidence": ["D10:2", "D2:1", "D2:4", "D1:3"],
        }
    )
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].evidence_session_ids == ["1", "10", "2"]


def test_evidence_without_colon_excluded_from_session_ids(tmp_path):
    rec = qa_only_record({"question": "q", "answer": "a", "evidence": ["D1:1", "x"]})
    sample = load_single(tmp_path, rec)
    assert sample.qa[0].evidence_turn_ids == ["D1:1", "x"]
    assert sample.qa[0].evidence_session_ids == ["1"]


def test_limit_caps_number_of_samples(tmp_path):
    second = record()
    second["sample_id"] = "conv-10"
    loader = make_loader(tmp_path, [record(), second])
    samples = loader.load(limit=1)
    assert [s.sample_id for s in samples] == ["conv-9"]


def test_empty_conversation_yields_bare_sample(tmp_path):
    sample = load_single(tmp_path, {"conversation": {}})
    assert sample.sample_id == "0"
    assert sample.conversation.speakers == []
    assert sample.conversation.sessions == []
    assert sample.qa == []


def test_pull_reuses_cached_file_without_network(tmp_path, monkeypatch):
    loader = make_loader(tmp_path, [])

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    path = loader.pull()
    assert path == loader.cache_dir / "locomo10.json"
    assert loader.load() == []


def test_pull_downloads_into_cache_when_missing(tmp_path, monkeypatch):
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append((url, Path(filename)))
        Path(filename).write_text("[]")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    url = "https://example.invalid/locomo.json"
    loader = LocomoLoader(cache_dir=tmp_path, url=url)
    path = loader.pull()
    assert path == loader.cache_dir / "locomo10.json"
    assert calls == [(url, path)]
    assert path.read_text() == "[]"


def test_unknown_variant_is_rejected(tmp_path):
    loader = make_loader(tmp_path, [record()])
    with pytest.raises(ValueError, match="unknown variant"):
        loader.load(variant="nope")
