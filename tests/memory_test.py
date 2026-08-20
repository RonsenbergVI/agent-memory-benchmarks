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

"""The naive BM25 baseline: ingest, rank, provenance, and its tool surface."""

from amb.contracts import Session, Turn
from amb.memory import NaiveBenchmark, NaiveIngestToolset, NaiveMemory


def _session() -> Session:
    """One session with three distinguishable turns."""
    return Session(
        session_id="s1",
        turns=[
            Turn(turn_id="t1", speaker="user", text="I adopted a dog named Biscuit"),
            Turn(turn_id="t2", speaker="assistant", text="Congrats on the new dog"),
            Turn(turn_id="t3", speaker="user", text="My favorite food is ramen"),
        ],
    )


def test_search_ranks_matching_turns_first() -> None:
    """The turn sharing the query's terms outscores unrelated turns."""
    memory = NaiveMemory()
    memory.ingest_session("c1", _session())
    hits = memory.search("c1", "dog Biscuit", k=3)
    assert hits
    assert "Biscuit" in hits[0].content
    for earlier, later in zip(hits, hits[1:]):
        assert earlier.score is not None
        assert later.score is not None
        assert earlier.score >= later.score


def test_hits_carry_turn_and_session_provenance() -> None:
    """Every hit cites the turn and session it came from — what scoring reads."""
    memory = NaiveMemory()
    memory.ingest_session("c1", _session())
    hit = memory.search("c1", "ramen", k=1)[0]
    assert hit.turn_ids == ["t3"]
    assert hit.session_ids == ["s1"]


def test_conversations_are_isolated() -> None:
    """One conversation's memories never surface in another's search."""
    memory = NaiveMemory()
    memory.ingest_session("c1", _session())
    assert memory.search("c2", "dog", k=3) == []


def test_teardown_drops_everything() -> None:
    """After teardown the store is empty and searches return nothing."""
    memory = NaiveMemory()
    memory.ingest_session("c1", _session())
    memory.teardown()
    assert memory.search("c1", "dog", k=3) == []
    assert memory.stats()["stored_turns"] == 0


def test_ingest_toolset_validates_citations() -> None:
    """The write tool stores only memories citing real turn ids."""
    benchmark = NaiveBenchmark()
    memory = benchmark.create_system()
    session = _session()
    toolset = benchmark.create_ingest_toolset(memory, "c1", session)
    assert isinstance(toolset, NaiveIngestToolset)
    assert toolset.store_memory("User has a dog named Biscuit", ["t1"]) == "stored"
    refused = toolset.store_memory("made up", ["nope"])
    assert refused.startswith("not stored")
    hits = memory.search("c1", "Biscuit", k=1)
    assert hits
    assert hits[0].turn_ids == ["t1"]


def test_benchmark_supports_agentic_mode() -> None:
    """The baseline declares both halves of its tool surface."""
    assert NaiveBenchmark.supports_agentic()
