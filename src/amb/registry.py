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

from importlib.metadata import EntryPoint, entry_points

from pydantic import BaseModel, ConfigDict

from amb.base import Benchmark, Memory
from amb.constants import ENTRY_POINT_GROUP


class BenchmarkSpec(BaseModel):
    """A registered integration, loaded lazily.

    Lazy loading means listing systems does not require every integration's
    SDK to be importable.
    """

    # EntryPoint is not a pydantic-aware type, so it is held as-is
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    entry_point: EntryPoint

    def benchmark_class(self) -> type[Benchmark]:
        """Load the entry point's Benchmark, wrapping a bare Memory.

        Raises:
            TypeError: if the entry point resolves to neither.
        """
        obj = self.entry_point.load()
        if isinstance(obj, type) and issubclass(obj, Benchmark):
            return obj
        if isinstance(obj, type) and issubclass(obj, Memory):
            return Benchmark.for_system(obj)
        raise TypeError(
            f"entry point {self.name!r} must resolve to a Benchmark or "
            f"Memory subclass, got {obj!r}"
        )

    def describe(self) -> str:
        """Return the system's one-line description."""
        return self.benchmark_class().system_class.description


def discover_benchmarks() -> dict[str, BenchmarkSpec]:
    """Find every integration registered under the entry-point group."""
    return {
        ep.name: BenchmarkSpec(name=ep.name, entry_point=ep)
        for ep in entry_points(group=ENTRY_POINT_GROUP)
    }


def get_benchmark(name: str) -> BenchmarkSpec:
    """Look up one registered integration by name.

    Raises:
        KeyError: if no integration is registered under that name.
    """
    specs = discover_benchmarks()
    if name not in specs:
        raise KeyError(f"unknown memory system {name!r}; available: {sorted(specs)}")
    return specs[name]
