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

"""An anchor extractor on Anthropic, for an SDK that ships only OpenAI's.

Plugs into fraise's own `Extractor` interface with the prompt and output
schema of the SDK's `OpenAIExtractor`, so only the provider differs. The
SDK keeps both private (0.2.x, pinned): re-check them on a bump, and drop
this module the day the SDK ships an Anthropic extractor of its own.
"""

import json

import anthropic
from fraise_sdk.errors import FraiseError
from fraise_sdk.providers.base import Anchor, Context, Extractor
from fraise_sdk.providers.openai import (
    _ANCHORS_SCHEMA,
    _EXTRACTION_SYSTEM_PROMPT,
    _MAX_COMPLETION_TOKENS,
)


class AnthropicExtractor(Extractor):
    """The topic and entity anchors of one message, from a Claude model."""

    def __init__(self, model: str) -> None:
        """Bind the model; the key comes from ANTHROPIC_API_KEY."""
        self._client = anthropic.Anthropic()
        self._model = model

    def extract(self, text: str, context: Context) -> list[Anchor]:
        """Return the topics and entities `text` is about, as anchors.

        Raises:
            FraiseError: on a refusal or an answer that is not the schema,
                which the client turns into a stored, untagged message.
        """
        payload = {
            "recent": list(context.recent),
            "vocabulary": {
                "topics": [a.value for a in context.vocabulary if a.type == "topic"],
                "entities": [a.value for a in context.vocabulary if a.type == "entity"],
            },
            "message": text,
        }
        response = self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_COMPLETION_TOKENS,
            system=_EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            output_config={
                "format": {"type": "json_schema", "schema": _ANCHORS_SCHEMA}
            },
        )
        answer = next((b.text for b in response.content if b.type == "text"), "")
        try:
            parsed = json.loads(answer)
            return [Anchor(value=v, type="topic") for v in parsed["topics"]] + [
                Anchor(value=v, type="entity") for v in parsed["entities"]
            ]
        except (TypeError, ValueError, KeyError) as exc:
            raise FraiseError(
                f"anchor extraction answer unparseable "
                f"(stop_reason={response.stop_reason})"
            ) from exc
