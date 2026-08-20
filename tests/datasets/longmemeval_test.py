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

"""Offline unit tests for the LongMemEval loader.

No network: load() is exercised through a subclass whose pull() returns a
handcrafted local JSON file, and pull() itself is exercised against fake
HfApi / hf_hub_download stand-ins patched into the module.
"""

import json
from pathlib import Path

import pytest

from amb.constants import Dataset
from amb.datasets import longmemeval
from amb.datasets.longmemeval import DEFAULT_REPO, LongMemEvalLoader


class _OfflineLoader(LongMemEvalLoader):
    """A LongMemEvalLoader whose pull() reads a local handcrafted file."""

    def __init__(self, data_path: Path, cache_dir: Path) -> None:
        super().__init__(cache_dir=cache_dir)
        self.data_path = data_path
        self.pulled_variants: list[str | None] = []

    def pull(self, variant: str | None = None) -> Path:
        self.pulled_variants.append(variant)
        return self.data_path


def _loader_for(tmp_path: Path, records: list[dict]) -> _OfflineLoader:
    data = tmp_path / "longmemeval_records.json"
    data.write_text(json.dumps(records))
    return _OfflineLoader(data, cache_dir=tmp_path / "cache")


def _patch_hub(monkeypatch, files, calls):
    class _FakeApi:
        def list_repo_files(self, repo, repo_type=None):
            calls.append(("list", repo, repo_type))
            return list(files)

    def _fake_download(repo, filename, repo_type=None, cache_dir=None):
        calls.append(("download", repo, filename, repo_type, cache_dir))
        return f"/hub-cache/{filename}"

    monkeypatch.setattr(longmemeval, "HfApi", _FakeApi)
    monkeypatch.setattr(longmemeval, "hf_hub_download", _fake_download)


def test_loader_metadata(tmp_path):
    loader = LongMemEvalLoader(cache_dir=tmp_path)
    assert loader.name is Dataset.LONGMEMEVAL
    assert loader.variants == ("s", "m", "oracle")
    assert loader.default_variant == "s"
    assert loader.repo == DEFAULT_REPO
    assert loader.cache_dir == tmp_path / "longmemeval"
    assert loader.cache_dir.is_dir()


def test_resolve_variant_defaults_to_s(tmp_path):
    assert LongMemEvalLoader(cache_dir=tmp_path).resolve_variant(None) == "s"


@pytest.mark.parametrize("variant", ["s", "m", "oracle"])
def test_resolve_variant_accepts_each_published_variant(tmp_path, variant):
    loader = LongMemEvalLoader(cache_dir=tmp_path)
    assert loader.resolve_variant(variant) == variant


def test_resolve_variant_rejects_unknown_name(tmp_path):
    loader = LongMemEvalLoader(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown variant 'xl'"):
        loader.resolve_variant("xl")


def test_load_yields_one_sample_per_question(tmp_path, longmemeval_rich_record):
    second = {**longmemeval_rich_record, "question_id": "q_2"}
    samples = _loader_for(tmp_path, [longmemeval_rich_record, second]).load()
    assert [s.sample_id for s in samples] == ["42", "q_2"]
    for sample in samples:
        assert len(sample.qa) == 1


def test_limit_caps_samples(tmp_path, longmemeval_rich_record):
    second = {**longmemeval_rich_record, "question_id": "q_2"}
    loader = _loader_for(tmp_path, [longmemeval_rich_record, second])
    assert [s.sample_id for s in loader.load(limit=1)] == ["42"]
    assert len(loader.load(limit=None)) == 2


def test_sample_identity(longmemeval_rich_sample):
    assert longmemeval_rich_sample.sample_id == "42"
    assert longmemeval_rich_sample.dataset == "longmemeval"
    assert longmemeval_rich_sample.conversation.conversation_id == "42"
    assert longmemeval_rich_sample.conversation.speakers == ["user", "assistant"]


def test_sessions_pair_ids_with_dates(longmemeval_rich_sample):
    sessions = longmemeval_rich_sample.conversation.sessions
    assert [s.session_id for s in sessions] == ["answer_s1", "noise_s2"]
    expected_dates = ["2023/05/01 (Mon) 10:00", "2023/05/03 (Wed) 18:45"]
    assert [s.timestamp for s in sessions] == expected_dates


def test_turn_ids_are_session_id_colon_index(longmemeval_rich_sample):
    first, second = longmemeval_rich_sample.conversation.sessions
    assert [t.turn_id for t in first.turns] == ["answer_s1:0", "answer_s1:1"]
    assert [t.turn_id for t in second.turns] == ["noise_s2:0", "noise_s2:1"]


def test_turns_keep_speaker_and_text(longmemeval_rich_sample):
    turn = longmemeval_rich_sample.conversation.sessions[0].turns[1]
    assert turn.speaker == "assistant"
    assert turn.text == "Congratulations on the cello!"


def test_qa_pair_fields(longmemeval_rich_sample):
    (qa,) = longmemeval_rich_sample.qa
    assert qa.question_id == "42"
    assert qa.question == "What instrument did the user pick up?"
    assert qa.answer == "1987"
    assert qa.category == "single-session-user"
    assert qa.question_date == "2023/05/20 (Sat) 02:21"


def test_answer_session_ids_become_evidence_session_ids(longmemeval_rich_sample):
    assert longmemeval_rich_sample.qa[0].evidence_session_ids == ["answer_s1"]


def test_only_has_answer_turns_become_evidence_turn_ids(longmemeval_rich_sample):
    assert longmemeval_rich_sample.qa[0].evidence_turn_ids == ["answer_s1:0"]


def test_evidence_collected_across_sessions_in_haystack_order(tmp_path):
    record = {
        "question_id": "multi",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["d1", "d2"],
        "answer_session_ids": ["s2", "s1", "s2"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "a", "has_answer": True},
                {"role": "assistant", "content": "b"},
            ],
            [
                {"role": "user", "content": "c", "has_answer": False},
                {"role": "assistant", "content": "d", "has_answer": True},
            ],
        ],
    }
    (sample,) = _loader_for(tmp_path, [record]).load()
    assert sample.qa[0].evidence_turn_ids == ["s1:0", "s2:1"]
    # duplicates collapse and the ids come back sorted
    assert sample.qa[0].evidence_session_ids == ["s1", "s2"]


def test_numeric_ids_are_coerced_to_str_and_sorted_lexicographically(tmp_path):
    record = {
        "question_id": 7,
        "haystack_session_ids": [10, 9],
        "answer_session_ids": [10, 9],
        "haystack_sessions": [[{"role": "user", "content": "x"}], []],
    }
    (sample,) = _loader_for(tmp_path, [record]).load()
    assert [s.session_id for s in sample.conversation.sessions] == ["10", "9"]
    assert sample.conversation.sessions[0].turns[0].turn_id == "10:0"
    assert sample.conversation.sessions[1].turns == []
    # sorted() on strings: "10" sorts before "9"
    assert sample.qa[0].evidence_session_ids == ["10", "9"]


def test_missing_ids_and_dates_fall_back_to_index_and_none(tmp_path):
    record = {
        "question_id": "fallback",
        "haystack_session_ids": ["only"],
        "haystack_dates": ["2023/05/01 (Mon) 10:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "named session"}],
            [{"role": "user", "content": "unnamed session"}],
        ],
    }
    (sample,) = _loader_for(tmp_path, [record]).load()
    second = sample.conversation.sessions[1]
    assert second.session_id == "1"
    assert second.timestamp is None
    assert second.turns[0].turn_id == "1:0"


def test_message_role_defaults_to_user_and_content_to_empty(tmp_path):
    record = {
        "question_id": "defaults",
        "haystack_session_ids": ["s"],
        "haystack_sessions": [[{}]],
    }
    (sample,) = _loader_for(tmp_path, [record]).load()
    turn = sample.conversation.sessions[0].turns[0]
    assert turn.speaker == "user"
    assert turn.text == ""


def test_minimal_record_defaults(tmp_path):
    (sample,) = _loader_for(tmp_path, [{"question_id": "bare"}]).load()
    assert sample.conversation.sessions == []
    (qa,) = sample.qa
    assert qa.question == ""
    assert qa.answer == ""
    assert qa.category is None
    assert qa.question_date is None
    assert qa.evidence_turn_ids == []
    assert qa.evidence_session_ids == []


def test_null_answer_becomes_the_string_none(tmp_path, longmemeval_rich_record):
    # Documents current behaviour: a JSON null answer goes through str() and
    # comes out as the literal string "None", not "" or None.
    record = {**longmemeval_rich_record, "answer": None}
    (sample,) = _loader_for(tmp_path, [record]).load()
    assert sample.qa[0].answer == "None"


def test_load_forwards_variant_to_pull(tmp_path, longmemeval_rich_record):
    loader = _loader_for(tmp_path, [longmemeval_rich_record])
    loader.load()
    loader.load("oracle")
    assert loader.pulled_variants == [None, "oracle"]


def test_pull_matches_variant_token_and_downloads_first_sorted(tmp_path, monkeypatch):
    calls = []
    files = [
        "README.md",
        "longmemeval_m.json",
        "longmemeval_oracle.json",
        "longmemeval_s.json",
        "longmemeval_s",
    ]
    _patch_hub(monkeypatch, files, calls)
    loader = LongMemEvalLoader(cache_dir=tmp_path, repo="someone/custom-repo")
    result = loader.pull("s")
    # both "_s" files match; sorted() puts the extensionless one first
    assert result == Path("/hub-cache/longmemeval_s")
    expected_download = (
        "download",
        "someone/custom-repo",
        "longmemeval_s",
        "dataset",
        tmp_path / "longmemeval",
    )
    assert calls == [("list", "someone/custom-repo", "dataset"), expected_download]


def test_pull_oracle_matches_oracle_token_not_markdown(tmp_path, monkeypatch):
    calls = []
    files = ["longmemeval_oracle.md", "longmemeval_oracle.json", "longmemeval_s.json"]
    _patch_hub(monkeypatch, files, calls)
    result = LongMemEvalLoader(cache_dir=tmp_path).pull("oracle")
    assert result == Path("/hub-cache/longmemeval_oracle.json")


def test_pull_raises_when_variant_file_missing(tmp_path, monkeypatch):
    _patch_hub(monkeypatch, ["longmemeval_s.json"], [])
    loader = LongMemEvalLoader(cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="variant 'm'"):
        loader.pull("m")


def test_pull_rejects_unknown_variant_before_touching_the_hub(tmp_path, monkeypatch):
    def _no_network(*args, **kwargs):
        raise AssertionError("the hub must not be reached for an unknown variant")

    monkeypatch.setattr(longmemeval, "HfApi", _no_network)
    monkeypatch.setattr(longmemeval, "hf_hub_download", _no_network)
    loader = LongMemEvalLoader(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown variant 'nope'"):
        loader.pull("nope")
