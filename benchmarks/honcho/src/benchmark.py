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

"""The Honcho benchmark object — what the entry point resolves to."""

from typing import ClassVar

from amb.base import Benchmark
from src.memory import HonchoMemory
from src.toolset import HonchoIngestToolset, HonchoSearchToolset


class HonchoBenchmark(Benchmark):
    """The benchmark object the `honcho` entry point resolves to.

    No spend tracker beyond the default: every model call happens inside
    the Honcho server, which reports no usage, so the system declares
    `usage_coverage = "none"` rather than publish an observed zero.
    """

    name: ClassVar[str] = "honcho"
    system_class = HonchoMemory
    search_toolset_class = HonchoSearchToolset
    ingest_toolset_class = HonchoIngestToolset
