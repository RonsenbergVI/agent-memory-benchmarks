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

"""Dataset base classes."""

from enum import StrEnum
from pathlib import Path

DEFAULT_DATA_DIR: Path = Path(".data")
ENTRY_POINT_GROUP: str = "amb.systems"
DEFAULT_REPORT_DIR: Path = Path("plots")


class Dataset(StrEnum):
    """The benchmark datasets this harness can load."""

    LOCOMO = "locomo"
    LONGMEMEVAL = "longmemeval"


class RunType(StrEnum):
    """The benchmark run type."""

    DIRECT = "direct"
    AGENT = "agent"


TOKEN_TRACKING_KEYS: tuple[str, ...] = (
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_calls",
    "embedding_tokens",
    "embedding_calls",
)

LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e1e0d9",
    "series": "#2a78d6",
    "categories": (
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ),
}

DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#2c2c2a",
    "series": "#3987e5",
    "categories": (
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ),
}
