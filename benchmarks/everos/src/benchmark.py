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

"""The EverOS benchmark object — what the entry point resolves to."""

from typing import ClassVar

from amb.base import Benchmark
from src.memory import EverOSMemory
from src.toolset import EverOSIngestToolset, EverOSSearchToolset


class EverOSBenchmark(Benchmark):
    """The benchmark object the `everos` entry point resolves to.

    The default `OpenAIUsageTracker` is the right one here: EverOS runs
    in-process and reaches its LLM and embedder through the openai SDK,
    so the tracker sees every call and the recorded spend is measured
    rather than inferred.
    """

    name: ClassVar[str] = "everos"
    system_class = EverOSMemory
    search_toolset_class = EverOSSearchToolset
    ingest_toolset_class = EverOSIngestToolset
