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

"""The Cognee benchmark object — what the entry point resolves to."""

from typing import ClassVar

from amb.base import Benchmark
from amb.callbacks import TiktokenUsageTracker
from src.memory import CogneeMemory
from src.toolset import CogneeIngestToolset, CogneeSearchToolset


class CogneeBenchmark(Benchmark):
    """The benchmark object the `cognee` entry point resolves to.

    Cognee reaches OpenAI through litellm rather than the openai SDK's
    own client methods, so the default `OpenAIUsageTracker` sees nothing
    and would record a cognee run as free. The adapter books its own
    spend from a litellm callback instead, and `TiktokenUsageTracker`
    reads those counters at the same lifecycle boundaries — the same
    arrangement letta uses for spend that happens inside its server.
    """

    name: ClassVar[str] = "cognee"
    system_class = CogneeMemory
    search_toolset_class = CogneeSearchToolset
    ingest_toolset_class = CogneeIngestToolset
    callback_classes = (TiktokenUsageTracker,)
