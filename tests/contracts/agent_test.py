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

from amb.contracts.agent import GenerationResult, Judgment


def test_judgment_stores_fields_verbatim():
    judgment = Judgment(correct=True, reasoning="answer matches the gold label")
    assert judgment.correct is True
    assert judgment.reasoning == "answer matches the gold label"


def test_judgment_correct_is_required():
    with pytest.raises(ValidationError, match="Field required") as exc_info:
        Judgment.model_validate({"reasoning": "no verdict given"})
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("correct",)


def test_judgment_reasoning_is_required():
    with pytest.raises(ValidationError, match="Field required") as exc_info:
        Judgment.model_validate({"correct": False})
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("reasoning",)


def test_judgment_missing_both_fields_reports_both_errors():
    with pytest.raises(ValidationError, match="2 validation errors") as exc_info:
        Judgment.model_validate({})
    locations = [error["loc"] for error in exc_info.value.errors()]
    assert locations == [("correct",), ("reasoning",)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), (1, True), (0, False)],
)
def test_judgment_correct_coerces_boolean_like_input(raw, expected):
    judgment = Judgment.model_validate({"correct": raw, "reasoning": "r"})
    assert judgment.correct is expected


def test_judgment_correct_rejects_non_boolean_input():
    with pytest.raises(ValidationError, match="valid boolean") as exc_info:
        Judgment.model_validate({"correct": "maybe", "reasoning": "r"})
    assert exc_info.value.errors()[0]["loc"] == ("correct",)


def test_judgment_reasoning_rejects_non_string_input():
    with pytest.raises(ValidationError, match="valid string") as exc_info:
        Judgment.model_validate({"correct": True, "reasoning": 123})
    errors = exc_info.value.errors()
    assert errors[0]["type"] == "string_type"
    assert errors[0]["loc"] == ("reasoning",)


def test_judgment_model_dump_round_trip():
    judgment = Judgment(correct=False, reasoning="dates disagree")
    dumped = judgment.model_dump()
    assert dumped == {"correct": False, "reasoning": "dates disagree"}
    assert Judgment.model_validate(dumped) == judgment


def test_judgment_extra_fields_are_ignored():
    judgment = Judgment.model_validate(
        {"correct": True, "reasoning": "r", "confidence": 0.9}
    )
    assert judgment.model_dump() == {"correct": True, "reasoning": "r"}


def test_generation_result_token_counts_default_to_zero():
    result = GenerationResult(text="Paris")
    assert result.text == "Paris"
    assert result.input_tokens == 0
    assert result.output_tokens == 0


def test_generation_result_stores_explicit_token_counts():
    result = GenerationResult(text="Paris", input_tokens=128, output_tokens=7)
    assert result.input_tokens == 128
    assert result.output_tokens == 7


def test_generation_result_text_is_the_only_required_field():
    with pytest.raises(ValidationError, match="1 validation error") as exc_info:
        GenerationResult.model_validate({})
    errors = exc_info.value.errors()
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"] == ("text",)


def test_generation_result_empty_text_is_allowed():
    result = GenerationResult(text="")
    assert result.text == ""


def test_generation_result_text_rejects_non_string_input():
    with pytest.raises(ValidationError, match="valid string") as exc_info:
        GenerationResult.model_validate({"text": None})
    assert exc_info.value.errors()[0]["loc"] == ("text",)


def test_generation_result_token_counts_coerce_numeric_strings():
    result = GenerationResult.model_validate(
        {"text": "t", "input_tokens": "42", "output_tokens": "0"}
    )
    assert result.input_tokens == 42
    assert result.output_tokens == 0


def test_generation_result_token_counts_reject_fractional_numbers():
    with pytest.raises(ValidationError, match="fractional part") as exc_info:
        GenerationResult.model_validate({"text": "t", "input_tokens": 2.5})
    errors = exc_info.value.errors()
    assert errors[0]["type"] == "int_from_float"
    assert errors[0]["loc"] == ("input_tokens",)


def test_generation_result_token_counts_reject_non_numeric_input():
    with pytest.raises(ValidationError, match="valid integer") as exc_info:
        GenerationResult.model_validate({"text": "t", "output_tokens": "many"})
    assert exc_info.value.errors()[0]["loc"] == ("output_tokens",)


def test_generation_result_negative_token_counts_are_accepted():
    # The contract puts no lower bound on token counts; document that.
    result = GenerationResult(text="t", input_tokens=-1)
    assert result.input_tokens == -1


def test_generation_result_model_dump_round_trip():
    result = GenerationResult(text="42", input_tokens=10, output_tokens=3)
    dumped = result.model_dump()
    assert dumped == {"text": "42", "input_tokens": 10, "output_tokens": 3}
    assert GenerationResult.model_validate(dumped) == result
