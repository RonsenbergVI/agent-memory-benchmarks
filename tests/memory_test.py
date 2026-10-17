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

import pytest

from amb.contracts import Conversation, QAPair, Sample, Session, Turn
from amb.memory import NaiveBenchmark, NaiveMemory
from amb.runner import RunConfig, Runner


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


def test_run_ingests_through_the_system_and_searches_through_tools() -> None:
    """Ingestion is the system's own ingest_session; the agent only searches."""
    sample = Sample(
        sample_id="sample-1",
        dataset="unit",
        conversation=Conversation(conversation_id="c1", sessions=[_session()]),
        qa=[
            QAPair(
                question_id="q1",
                question="What is the dog called?",
                answer="Biscuit",
                evidence_turn_ids=["t1"],
            )
        ],
    )
    config = RunConfig(dataset="unit", model="test")
    run = Runner(NaiveBenchmark(), config, samples=[sample]).run()
    assert run.ingestion_records[0].system_stats["stored_turns"] == 3
    row = run.question_records[0]
    assert row.error is None
    assert row.predicted_answer is not None
    assert row.num_searches
    assert row.num_searches >= 1


def test_model_ids_need_a_provider() -> None:
    """A bare model name is ambiguous across providers, so it is refused."""
    from amb.base import split_model

    assert split_model("anthropic:claude-haiku-5-5") == (
        "anthropic",
        "claude-haiku-5-5",
    )
    with pytest.raises(ValueError, match="provider:name"):
        split_model("gpt-5-mini")


def test_model_name_refuses_an_unsupported_provider() -> None:
    """A system never silently runs a provider it does not support natively."""

    class OpenAIOnly(NaiveMemory):
        llm_providers = ("openai",)

    assert OpenAIOnly.model_name("openai:gpt-5-mini", "llm") == "gpt-5-mini"
    with pytest.raises(ValueError, match="cannot run its llm on 'anthropic'"):
        OpenAIOnly.model_name("anthropic:claude-haiku-5-5", "llm")
    with pytest.raises(ValueError, match="cannot run without an llm"):
        OpenAIOnly.model_name(None, "llm")
