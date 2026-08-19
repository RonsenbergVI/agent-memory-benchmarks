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

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel


class Heading(BaseModel):
    """A heading at `level` (1 = document title)."""

    level: int
    text: str


class Paragraph(BaseModel):
    """A block of prose."""

    text: str


class Table(BaseModel):
    """A table of already-formatted cells."""

    header: Sequence[str]
    rows: Sequence[Sequence[str]]


class Figure(BaseModel):
    """An image, linked at `path` relative to the document."""

    alt: str
    path: Path


class Rule(BaseModel):
    """A divider between two sibling sections."""


Block = Heading | Paragraph | Table | Figure | Rule
