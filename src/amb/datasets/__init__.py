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

from pathlib import Path

from amb.base import DatasetLoader
from amb.constants import Dataset
from amb.datasets.locomo import LocomoLoader
from amb.datasets.longmemeval import LongMemEvalLoader

LOADERS: dict[Dataset, type[DatasetLoader]] = {
    Dataset.LOCOMO: LocomoLoader,
    Dataset.LONGMEMEVAL: LongMemEvalLoader,
}


def get_loader(name: str | Dataset, cache_dir: Path | None = None) -> DatasetLoader:
    """Build the loader for a dataset, pointed at the given cache."""
    return LOADERS[Dataset(name)](cache_dir)
