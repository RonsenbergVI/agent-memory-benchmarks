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
from collections import Counter
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
    assert loader.variants == ("s", "m", "oracle", "xs-shared")
    assert loader.default_variant == "s"
    assert loader.repo == DEFAULT_REPO
    assert loader.cache_dir == tmp_path / "longmemeval"
    assert loader.cache_dir.is_dir()


def test_resolve_variant_defaults_to_s(tmp_path):
    assert LongMemEvalLoader(cache_dir=tmp_path).resolve_variant(None) == "s"


@pytest.mark.parametrize("variant", ["s", "m", "oracle", "xs-shared"])
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
    assert loader.pulled_variants == ["s", "oracle"]


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


def test_pull_xs_shared_downloads_the_s_file(tmp_path, monkeypatch):
    calls = []
    files = ["longmemeval_m.json", "longmemeval_oracle.json", "longmemeval_s.json"]
    _patch_hub(monkeypatch, files, calls)
    result = LongMemEvalLoader(cache_dir=tmp_path).pull("xs-shared")
    assert result == Path("/hub-cache/longmemeval_s.json")


def _record(qid, evidence, fillers, question=None):
    """A record whose haystack interleaves evidence and filler sessions by date.

    Session ids carry their day, so `fillers=[(3, "f3")]` is session "f3" dated
    2023/05/03; evidence sessions open with a has_answer turn.
    """
    sessions = sorted(
        [(day, sid, True) for day, sid in evidence]
        + [(day, sid, False) for day, sid in fillers],
        reverse=True,
    )
    return {
        "question_id": qid,
        "question": question or f"question {qid}?",
        "answer": f"answer {qid}",
        "question_type": "multi-session",
        "question_date": "2023/06/30 (Fri) 12:00",
        "answer_session_ids": [sid for _, sid in evidence],
        "haystack_session_ids": [sid for _, sid, _ in sessions],
        "haystack_dates": [f"2023/05/{day:02d} (Mon) 10:00" for day, _, _ in sessions],
        "haystack_sessions": [
            [
                {"role": "user", "content": sid, "has_answer": answer},
                {"role": "assistant", "content": "ok"},
            ]
            for _, sid, answer in sessions
        ],
    }


def _session_ids(sample):
    return [s.session_id for s in sample.conversation.sessions]


def _fillers(prefix, n, first_day=1):
    return [(first_day + i, f"{prefix}{i}") for i in range(n)]


def _typed_records(n_multi, n_temporal, n_abs):
    kinds = (
        [("multi-session", "")] * n_multi
        + [("temporal-reasoning", "")] * n_temporal
        + [("temporal-reasoning", "_abs")] * n_abs
    )
    records = []
    for i, (kind, suffix) in enumerate(kinds):
        record = _record(f"q{i}{suffix}", [(1, f"ev{i}")], [])
        record["question_type"] = kind
        records.append(record)
    return records


def test_xs_shared_draws_150_questions_stratified_by_type_and_abstention(tmp_path):
    loader = _loader_for(tmp_path, _typed_records(300, 180, 20))
    qas = [qa for sample in loader.load("xs-shared") for qa in sample.qa]
    kinds = Counter((qa.category, qa.question_id.endswith("_abs")) for qa in qas)
    assert kinds == {
        ("multi-session", False): 90,
        ("temporal-reasoning", False): 54,
        ("temporal-reasoning", True): 6,
    }


def test_xs_shared_draws_the_same_questions_every_load(tmp_path):
    loader = _loader_for(tmp_path, _typed_records(300, 180, 20))
    first = [qa.question_id for s in loader.load("xs-shared") for qa in s.qa]
    assert first == [qa.question_id for s in loader.load("xs-shared") for qa in s.qa]


def _shared_records(n, wordings=None):
    return [
        _record(
            f"q{i}",
            [(1 + i % 28, f"ev{i}")],
            _fillers(f"f{i}_", 30),
            question=(wordings or {}).get(i),
        )
        for i in range(n)
    ]


def test_xs_shared_deals_questions_round_robin_into_three_users(tmp_path):
    samples = _loader_for(tmp_path, _shared_records(25)).load("xs-shared")
    assert [s.sample_id for s in samples] == ["user_0", "user_1", "user_2"]
    assert [q.question_id for q in samples[0].qa][:3] == ["q0", "q3", "q6"]
    assert [len(s.qa) for s in samples] == [9, 8, 8]
    assert samples[0].conversation.conversation_id == "user_0"


def test_xs_shared_history_is_evidence_plus_twenty_fillers(tmp_path):
    samples = _loader_for(tmp_path, _shared_records(25)).load("xs-shared")
    ids = _session_ids(samples[0])
    assert {f"ev{i}" for i in range(0, 25, 3)} <= set(ids)
    assert len(ids) == 9 + 20
    dates = [s.timestamp or "" for s in samples[0].conversation.sessions]
    assert dates == sorted(dates)


def test_xs_shared_keeps_each_questions_own_labels(tmp_path):
    samples = _loader_for(tmp_path, _shared_records(25)).load("xs-shared")
    qa = samples[0].qa[1]
    assert qa.evidence_session_ids == ["ev3"]
    assert qa.evidence_turn_ids == ["ev3:0"]


def test_xs_shared_deduplicates_sessions_across_questions(tmp_path):
    shared = _fillers("common", 30)
    records = [_record(f"q{i}", [(1, f"ev{i}")], shared) for i in range(20)]
    samples = _loader_for(tmp_path, records).load("xs-shared")
    ids = _session_ids(samples[0])
    assert len(ids) == len(set(ids)) == 7 + 20


def test_xs_shared_fillers_are_seeded_by_user(tmp_path):
    shared = _fillers("common", 30)
    records = [_record(f"q{i}", [(1, f"ev{i}")], shared) for i in range(20)]
    loader = _loader_for(tmp_path, records)
    first = [_session_ids(s) for s in loader.load("xs-shared")]
    assert first == [_session_ids(s) for s in loader.load("xs-shared")]
    assert {i for i in first[0] if i.startswith("common")} != {
        i for i in first[1] if i.startswith("common")
    }


def test_xs_shared_never_puts_one_wording_twice_on_a_user(tmp_path):
    records = _shared_records(6, {0: "same?", 3: "same?"})
    samples = _loader_for(tmp_path, records).load("xs-shared")
    assert [q.question_id for q in samples[0].qa] == ["q0", "q5"]
    assert [q.question_id for q in samples[1].qa] == ["q1", "q3"]
    assert [q.question_id for q in samples[2].qa] == ["q2", "q4"]


def test_xs_shared_refuses_a_wording_asked_more_than_three_times(tmp_path):
    records = _shared_records(4, dict.fromkeys(range(4), "same?"))
    loader = _loader_for(tmp_path, records)
    with pytest.raises(ValueError, match="share the wording 'same\\?'"):
        loader.load("xs-shared")


def test_xs_shared_abstention_questions_keep_their_own_history(tmp_path):
    records = _shared_records(3)
    records.insert(1, _record("q9_abs", [(3, "ev_abs")], _fillers("fa", 9, 6)))
    samples = _loader_for(tmp_path, records).load("xs-shared")
    assert [s.sample_id for s in samples] == ["user_0", "user_1", "user_2", "q9_abs"]
    alone = samples[-1]
    assert [q.question_id for q in alone.qa] == ["q9_abs"]
    ids = _session_ids(alone)
    assert "ev_abs" in ids
    assert len(ids) == 1 + 5
    assert all("ev_abs" not in _session_ids(s) for s in samples[:-1])


def test_xs_shared_limit_counts_histories(tmp_path):
    samples = _loader_for(tmp_path, _shared_records(25)).load("xs-shared", limit=2)
    assert [s.sample_id for s in samples] == ["user_0", "user_1"]
    assert len(samples[0].qa) == 9
