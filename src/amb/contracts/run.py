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

from pydantic import BaseModel, Field


class QuestionRecord(BaseModel):
    """One question's raw observations — no scores, raw data only."""

    sample_id: str
    question_id: str
    question: str | None = None
    category: str | None = None
    gold_answer: str | None = None
    # retrieval observations
    search_s: float | None = None
    num_hits: int | None = None
    retrieved_session_ids: list[str] | None = None
    evidence_session_ids: list[str] | None = None
    retrieved_turn_ids: list[str] | None = None
    evidence_turn_ids: list[str] | None = None
    # answer observations (whenever an answer model ran)
    predicted_answer: str | None = None
    answer_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    num_searches: int | None = None
    memory_tokens: int | None = None
    judge_correct: bool | None = None
    judge_reasoning: str | None = None
    error: str | None = None


class IngestionRecord(BaseModel):
    """One sample's ingestion (and phase-timing) observations."""

    sample_id: str
    ingest_s: float
    num_turns: int
    num_sessions: int | None = None
    system_stats: dict = Field(default_factory=dict)
    conversation_s: float | None = None
    questions_dropped: int | None = None
    reused: bool | None = None
    token_usage: dict | None = None
    # agentic mode: what the ingesting agent did and spent
    num_writes: int | None = None
    write_s: float | None = None
    agent_input_tokens: int | None = None
    agent_output_tokens: int | None = None


class Run(BaseModel):
    """Everything one system x dataset run observed."""

    run_id: str
    system: str
    dataset: str
    # models and --param overrides join the run's identity, so variants coexist as rows
    ingestion_model: str | None = None
    embedding_model: str | None = None
    # "full" | "partial" | "none": how much of the system's spend this run accounts for
    usage_coverage: str = "full"
    system_params: dict = Field(default_factory=dict)
    variant: str | None = None
    mode: str = "direct"  # "direct" (harness-driven) or "agentic" (model-driven)
    k: int = 10
    model: str | None = None
    judge_model: str | None = None
    # set when ingestion was truncated: scores not comparable to a full run
    max_turns: int | None = None
    sample_seed: int | None = None
    # latency under N-way contention is a different experiment from single-tenant,
    # so the worker count joins the run's identity (1 = historical default)
    workers: int = 1
    system_version: str | None = None
    question_records: list[QuestionRecord] = Field(default_factory=list)
    ingestion_records: list[IngestionRecord] = Field(default_factory=list)

    def add_question(self, row: dict | QuestionRecord) -> QuestionRecord:
        """Validate and store one question's records."""
        record = QuestionRecord.model_validate(row)
        self.question_records.append(record)
        return record

    def add_ingestion(self, stats: dict | IngestionRecord) -> IngestionRecord:
        """Validate and store one sample's ingestion records."""
        record = IngestionRecord.model_validate(stats)
        self.ingestion_records.append(record)
        return record
