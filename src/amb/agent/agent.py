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
from pydantic_ai.models import Model, infer_model

from amb.agent.prompts import ANSWER_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT
from amb.agent.toolset import SearchToolset
from amb.constants import HARNESS_CLIENT_ATTR
from amb.contracts import GenerationResult, Judgment


def harness_model(model: str | Model) -> Model:
    """Resolve an agent model and mark its SDK client as amb's own.

    The usage tracker skips marked clients: the agent's tokens are reported
    on the row (`input_tokens`/`output_tokens`), never as memory spend.
    """
    resolved = infer_model(model) if isinstance(model, str) else model
    client = getattr(resolved, "client", None)
    if client is not None:
        setattr(client, HARNESS_CLIENT_ATTR, True)
    return resolved


def answer_with_memory(
    question: str,
    toolset: SearchToolset,
    model: str | Model,
    question_date: str | None = None,
) -> GenerationResult:
    """Answer one question, the model searching the system's own tools.

    Searches run through the system's own tools and are recorded on the toolset.
    """
    agent = Agent(
        harness_model(model),
        system_prompt=ANSWER_SYSTEM_PROMPT,
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


def judge_answer(
    question: str, prediction: str, gold: str, model: str | Model
) -> Judgment:
    """Grade a predicted answer against the gold answer (evaluation phase)."""
    agent = Agent(
        harness_model(model),
        system_prompt=JUDGE_SYSTEM_PROMPT,
        output_type=Judgment,
    )
    result = agent.run_sync(
        f"Question: {question}\nGold answer: {gold}\nPredicted answer: {prediction}"
    )
    return result.output
