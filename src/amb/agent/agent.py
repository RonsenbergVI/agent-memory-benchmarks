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

from pydantic_ai import Agent
from pydantic_ai.models import Model

from amb.agent.helpers import format_context, format_session
from amb.agent.prompts import (
    AGENTIC_ANSWER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    INGEST_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)
from amb.agent.toolset import IngestToolset, SearchToolset
from amb.contracts import GenerationResult, Judgment, MemoryHit, Session


def answer_question(
    question: str,
    hits: list[MemoryHit],
    model: str | Model,
    question_date: str | None = None,
) -> GenerationResult:
    """Answer one question from pre-retrieved memory context (direct mode)."""
    agent = Agent(model, system_prompt=ANSWER_SYSTEM_PROMPT)
    prompt = f"Memory context:\n{format_context(hits)}\n\n"
    if question_date:
        prompt += f"Today's date: {question_date}\n"
    prompt += f"Question: {question}"
    result = agent.run_sync(prompt)
    usage = result.usage
    return GenerationResult(
        text=result.output.strip(),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


def answer_with_memory(
    question: str,
    toolset: SearchToolset,
    model: str | Model,
    question_date: str | None = None,
) -> GenerationResult:
    """Answer one question with the model driving retrieval (agentic mode).

    Searches run through the system's own tools and are recorded on the toolset.
    """
    agent = Agent(
        model,
        system_prompt=AGENTIC_ANSWER_SYSTEM_PROMPT,
        toolsets=[toolset],
    )
    prompt = ""
    if question_date:
        prompt += f"Today's date: {question_date}\n"
    prompt += f"Question: {question}"
    result = agent.run_sync(prompt)
    usage = result.usage
    return GenerationResult(
        text=result.output.strip(),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


def ingest_with_agent(
    session: Session,
    toolset: IngestToolset,
    model: str | Model,
) -> GenerationResult:
    """Store one session with the model driving the writes (agentic mode).

    The model decides what to store; writes are recorded on the toolset.
    """
    agent = Agent(
        model,
        system_prompt=INGEST_SYSTEM_PROMPT,
        toolsets=[toolset],
    )
    result = agent.run_sync(format_session(session))
    usage = result.usage
    return GenerationResult(
        text=result.output.strip(),
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
    )


def judge_answer(
    question: str, prediction: str, gold: str, model: str | Model
) -> Judgment:
    """Grade a predicted answer against the gold answer (evaluation phase)."""
    agent = Agent(
        model,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        output_type=Judgment,
    )
    result = agent.run_sync(
        f"Question: {question}\nGold answer: {gold}\nPredicted answer: {prediction}"
    )
    return result.output
