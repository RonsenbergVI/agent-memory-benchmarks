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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from amb.constants import DEFAULT_DATA_DIR, Dataset
from amb.contracts import Sample


class DatasetLoader(ABC):
    """Downloads a benchmark's raw files and normalizes them.

    Subclasses turn provider-specific payloads into list[Sample].
    """

    name: ClassVar[Dataset]
    variants: ClassVar[tuple[str, ...]] = ("default",)
    default_variant: ClassVar[str] = "default"

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Point the loader at its cache directory, creating it if needed."""
        self.cache_dir = (cache_dir or DEFAULT_DATA_DIR) / self.name.value
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve_variant(self, variant: str | None) -> str:
        """Return the variant to use, rejecting names this dataset lacks.

        Raises:
            ValueError: if the variant is not one of `variants`.
        """
        variant = variant or self.default_variant
        if variant not in self.variants:
            raise ValueError(
                f"unknown variant {variant!r} for {self.name}; "
                f"expected one of {self.variants}"
            )
        return variant

    @abstractmethod
    def pull(self, variant: str | None = None) -> Path:
        """Download the raw data into the cache and return its local path.

        Idempotent — reuses the cached copy when present.
        """

    @abstractmethod
    def load(
        self, variant: str | None = None, limit: int | None = None
    ) -> list[Sample]:
        """Return normalized samples, pulling raw data first if needed.

        `limit` caps the number of samples, for smoke runs.
        """
