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

from typing import TYPE_CHECKING, ClassVar

from amb.base import Callback, Memory
from amb.callbacks import OpenAIUsageTracker
from amb.contracts import Sample, Session

if TYPE_CHECKING:
    from amb.agent.toolset import IngestToolset, SearchToolset


class Benchmark:
    """One memory system, declared for the harness to run.

    Subclass it, set `name` and `system_class`, and override the factory
    hooks below when the system needs objects built its own way.
    """

    name: ClassVar[str] = "unnamed"
    system_class: ClassVar[type[Memory]]
    default_params: ClassVar[dict] = {}
    # the system's own tool surface, used in agentic mode; a system supports
    # agentic mode only when its Benchmark sets both
    search_toolset_class: ClassVar[type | None] = None
    ingest_toolset_class: ClassVar[type | None] = None
    # Callback classes instantiated per run. Every benchmark gets the token
    # tracker: it measures in-process openai traffic always, and a counting
    # proxy's (AMB_USAGE_PROXY_URL) on top for server-backed systems —
    # a system that spends nothing records a truthful zero
    callback_classes: ClassVar[tuple[type[Callback], ...]] = (OpenAIUsageTracker,)

    def __init__(self, params: dict | None = None) -> None:
        """Hold the caller's system parameters (the CLI's `--param`).

        Only the explicit overrides are kept here — `default_params` is
        merged in at `create_system` — so the run records exactly what was
        asked for as part of its identity.
        """
        self.params = dict(params or {})

    @classmethod
    def supports_agentic(cls) -> bool:
        """Whether this system exposes both halves of its tool surface."""
        return bool(cls.search_toolset_class and cls.ingest_toolset_class)

    def create_memory(self) -> Memory:
        """Build the memory system, applying any `--param` overrides."""
        return self.system_class(**{**self.default_params, **self.params})

    def create_search_toolset(
        self, system: Memory, conversation_id: str, k: int = 10
    ) -> "SearchToolset":
        """Build the system's search toolset, per question (agentic mode)."""
        assert self.search_toolset_class is not None  # checked by Runner.run()
        return self.search_toolset_class(system, conversation_id, k=k)

    def create_ingest_toolset(
        self, system: Memory, conversation_id: str, session: Session
    ) -> "IngestToolset":
        """Build the system's write toolset, per session (agentic mode)."""
        assert self.ingest_toolset_class is not None  # checked by Runner.run()
        return self.ingest_toolset_class(system, conversation_id, session)

    def before_sample(self, system: Memory, sample: Sample) -> None:
        """Called after setup, before ingestion."""

    def after_sample(self, system: Memory, sample: Sample) -> None:
        """Called after all questions, before teardown."""

    @classmethod
    def for_system(cls, system_cls: type[Memory]) -> type["Benchmark"]:
        """Build a default Benchmark for a bare Memory.

        Used when an integration registered only a Memory subclass.
        """
        return type(
            f"{system_cls.__name__}Benchmark",
            (cls,),
            {"name": system_cls.name, "system_class": system_cls},
        )
