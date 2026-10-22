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


import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from amb.agent.toolset import MemoryToolset
from amb.base import Benchmark, spanned_turns
from amb.contracts import Conversation, MemoryHit, QAPair, Sample, Session, Turn
from amb.runner import RunConfig, Runner

BENCHMARKS = Path(__file__).resolve().parent.parent / "benchmarks"


def _session(session_id: str = "s1") -> Session:
    """Three turns, rendered by the adapters as `speaker: text` lines."""
    return Session(
        session_id=session_id,
        turns=[
            Turn(turn_id=f"{session_id}:1", speaker="Ann", text="I adopted a dog"),
            Turn(turn_id=f"{session_id}:2", speaker="Bo", text="What is its name?"),
            Turn(turn_id=f"{session_id}:3", speaker="Ann", text="Biscuit"),
        ],
    )


def _adapter(name: str) -> ModuleType:
    """An integration's memory module, loaded by path: every one is `src`."""
    spec = importlib.util.spec_from_file_location(
        f"amb_test_{name}_memory", BENCHMARKS / name / "src" / "memory.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- the harness ---------------------------------------------------------


def test_spanned_turns_places_a_slice_of_the_transcript():
    session = _session()
    assert spanned_turns(session, "Bo: What is its name?\nAnn: Biscuit") == [
        "s1:2",
        "s1:3",
    ]
    # a chunk cut mid-turn still covers that turn
    assert spanned_turns(session, "a dog\nBo: What") == ["s1:1", "s1:2"]
    assert spanned_turns(session, "\nAnn: Biscuit\n") == ["s1:3"]


def test_spanned_turns_refuses_text_that_is_not_a_slice():
    assert spanned_turns(_session(), "Ann adopted a dog") == []
    assert spanned_turns(_session(), "  ") == []


def test_default_resolution_expands_a_hit_to_its_sessions_turns(fake_memory_class):
    memory = fake_memory_class()
    hits = [
        MemoryHit(content="a", session_ids=["s2", "s2"]),
        MemoryHit(content="b", session_ids=["s1"], turn_ids=["s1:3"]),
        MemoryHit(content="c"),
    ]
    resolved = memory.resolve_turns(hits, [_session("s1"), _session("s2")])
    assert resolved[0].turn_ids == ["s2:1", "s2:2", "s2:3"]
    # a system's own turn ids are kept, never widened
    assert resolved[1].turn_ids == ["s1:3"]
    # no provenance stays none: the question drops out instead of scoring 0.0
    assert resolved[2].turn_ids == []
    assert memory.turn_provenance == "session"


def test_turns_are_resolved_outside_the_timed_search(fake_memory_class):
    class SlowToResolve(fake_memory_class):
        def search(self, conversation_id, query, k=10):
            return [MemoryHit(content="Ann adopted a dog", session_ids=["s1"])]

        def resolve_turns(self, hits, sessions):
            time.sleep(0.3)
            return super().resolve_turns(hits, sessions)

    sample = Sample(
        sample_id="sample-1",
        dataset="unit",
        conversation=Conversation(conversation_id="c1", sessions=[_session()]),
        qa=[
            QAPair(
                question_id="q1",
                question="What is the dog called?",
                answer="Biscuit",
                evidence_turn_ids=["s1:3"],
                evidence_session_ids=["s1"],
            )
        ],
    )
    benchmark = Benchmark.for_system(SlowToResolve)()
    run = Runner(benchmark, RunConfig(dataset="unit", model="test"), [sample]).run()
    row = run.question_records[0]
    assert row.error is None
    assert row.search_s is not None
    assert row.search_s < 0.3
    assert set(row.retrieved_turn_ids or []) == {"s1:1", "s1:2", "s1:3"}
    assert run.turn_provenance == "session"


def test_the_agent_never_sees_provenance(fake_memory_class):
    class Cited(fake_memory_class):
        def search(self, conversation_id, query, k=10):
            return [
                MemoryHit(
                    content="Biscuit",
                    score=1.0,
                    turn_ids=["s1:3"],
                    session_ids=["s1"],
                    metadata={"sources": ["Ann: Biscuit"], "chunk_ids": ["x"]},
                )
            ]

    toolset = MemoryToolset(Cited(), "c1")
    assert toolset.search_memory("dog") == [{"content": "Biscuit", "score": 1.0}]


def test_turn_provenance_survives_a_save_and_reload(tmp_path):
    from amb.contracts import Run
    from amb.reporting.run import RunReport

    run = Run(run_id="r1", system="alpha", dataset="unit", turn_provenance="chunk")
    saved = RunReport(run).save(tmp_path)
    assert json.loads((saved / "summary.json").read_text())["turn_provenance"] == (
        "chunk"
    )
    assert RunReport.load(saved).run.turn_provenance == "chunk"


# -- the integrations: narrowing from what each system already returned ---


def test_memos_narrows_a_memory_to_its_source_window():
    module = _adapter("memos")
    memory = object.__new__(module.MemOSMemory)
    window = MemoryHit(
        content="Ann's dog is Biscuit",
        session_ids=["s1"],
        metadata={"sources": ["Bo: What is its name?", "Ann: Biscuit"]},
    )
    structural = MemoryHit(content="topic", session_ids=["s1"], metadata={})
    resolved = memory.resolve_turns([window, structural], [_session()])
    assert resolved[0].turn_ids == ["s1:2", "s1:3"]
    # nothing to narrow by: the whole session, the coarser truth
    assert resolved[1].turn_ids == ["s1:1", "s1:2", "s1:3"]
    assert memory.turn_provenance == "window"


def test_memos_reads_sources_as_models_or_dicts():
    module = _adapter("memos")
    metadata = SimpleNamespace(
        sources=[SimpleNamespace(content="Ann: Biscuit"), {"content": "Bo: hi"}, {}]
    )
    assert module.MemOSMemory._source_texts(metadata) == ["Ann: Biscuit", "Bo: hi"]


def test_cognee_narrows_a_chunk_to_the_turns_it_spans():
    module = _adapter("cognee")
    memory = object.__new__(module.CogneeMemory)
    chunk = MemoryHit(content="Ann: I adopted a dog\nBo: What", session_ids=["s1"])
    rewritten = MemoryHit(content="Ann has a dog", session_ids=["s1"])
    resolved = memory.resolve_turns([chunk, rewritten], [_session()])
    assert resolved[0].turn_ids == ["s1:1", "s1:2"]
    assert resolved[1].turn_ids == ["s1:1", "s1:2", "s1:3"]
    assert memory.turn_provenance == "chunk"


def test_hindsight_narrows_a_memory_to_its_chunks():
    module = _adapter("hindsight")
    memory = object.__new__(module.HindsightMemory)
    memory._documents = {"c1:s1:0": "s1"}
    memory._chunks = {
        "k1": SimpleNamespace(document_id="c1:s1:0", chunk_text="Ann: Biscuit")
    }
    hit = MemoryHit(content="dog", session_ids=["s1"], metadata={"chunk_ids": ["k1"]})
    assert memory.resolve_turns([hit], [_session()])[0].turn_ids == ["s1:3"]
    assert memory.turn_provenance == "chunk"


def test_hindsight_keeps_the_session_when_a_chunk_cannot_be_placed(monkeypatch):
    module = _adapter("hindsight")
    memory = object.__new__(module.HindsightMemory)
    memory._documents = {"c1:s1:0": "s1"}

    def chunk(chunk_id):
        if chunk_id == "k1":
            return SimpleNamespace(document_id="c1:s1:0", chunk_text="Ann: Biscuit")
        raise LookupError(chunk_id)

    monkeypatch.setattr(memory, "_chunk", chunk)
    hit = MemoryHit(
        content="dog", session_ids=["s1"], metadata={"chunk_ids": ["k1", "k2"]}
    )
    assert memory.resolve_turns([hit], [_session()])[0].turn_ids == [
        "s1:1",
        "s1:2",
        "s1:3",
    ]


def test_hindsight_finds_an_observations_chunks_through_its_source_facts():
    module = _adapter("hindsight")
    world = SimpleNamespace(chunk_id="k1")
    observation = SimpleNamespace(chunk_id=None, source_fact_ids=["f1", "f2", "f3"])
    facts = {
        "f1": SimpleNamespace(chunk_id="k2"),
        "f2": SimpleNamespace(chunk_id="k2"),
        "f3": SimpleNamespace(chunk_id="k3"),
    }
    assert module.HindsightMemory._chunks_of(world, facts) == ["k1"]
    assert module.HindsightMemory._chunks_of(observation, facts) == ["k2", "k3"]


def test_everos_narrows_an_episode_to_its_memcell(monkeypatch):
    module = _adapter("everos")
    episodes = {
        "e1": SimpleNamespace(parent_type="memcell", parent_id="m1"),
        "e2": SimpleNamespace(parent_type="cluster", parent_id="c9"),
    }
    cell = SimpleNamespace(
        memcell_id="m1",
        session_id="s1",
        message_ids_json=json.dumps(["m_s1_1000_001", "m_s1_2000_002"]),
    )

    async def get_by_id(episode_id):
        return episodes.get(episode_id)

    async def find_by_ids(memcell_ids):
        return [cell] if "m1" in memcell_ids else []

    episode = ModuleType("everos.infra.persistence.lancedb.repos.episode")
    setattr(episode, "episode_repo", SimpleNamespace(get_by_id=get_by_id))
    memcell = ModuleType("everos.infra.persistence.sqlite.repos.memcell")
    setattr(memcell, "memcell_repo", SimpleNamespace(find_by_ids=find_by_ids))
    monkeypatch.setitem(sys.modules, episode.__name__, episode)
    monkeypatch.setitem(sys.modules, memcell.__name__, memcell)
    monkeypatch.setattr(module, "_await", asyncio.run)
    memory = object.__new__(module.EverOSMemory)
    hits = [
        MemoryHit(content="a", session_ids=["s1"], metadata={"episode_id": "e1"}),
        MemoryHit(content="b", session_ids=["s1"], metadata={"episode_id": "e2"}),
    ]
    resolved = memory.resolve_turns(hits, [_session()])
    assert resolved[0].turn_ids == ["s1:2", "s1:3"]
    # a merged episode has no memcell: the whole session
    assert resolved[1].turn_ids == ["s1:1", "s1:2", "s1:3"]
    assert memory.turn_provenance == "segment"


@pytest.mark.parametrize(
    ("name", "cls", "unit"),
    [
        ("fraise", "FraiseMemory", "turn"),
        ("graphiti", "GraphitiMemory", "session"),
        ("mem0", "Mem0Memory", "session"),
        ("langmem", "LangMemMemory", "session"),
    ],
)
def test_every_integration_declares_its_provenance(name, cls, unit):
    assert getattr(_adapter(name), cls).turn_provenance == unit
