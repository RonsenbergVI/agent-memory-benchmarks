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

import click

from amb.base.dataset import DEFAULT_DATA_DIR
from amb.datasets import LOADERS, get_loader


@click.group()
def datasets() -> None:
    """Inspect and fetch the benchmark datasets."""


@datasets.command(name="ls")
def ls() -> None:
    """List datasets and their variants."""
    for dataset, loader_cls in LOADERS.items():
        variants = ", ".join(loader_cls.variants)
        click.echo(
            f"{dataset.value:<12} variants: {variants} "
            f"(default: {loader_cls.default_variant})"
        )


@datasets.command()
@click.argument("dataset")
@click.option("--variant", help="dataset variant; defaults to the loader's own")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATA_DIR,
    show_default=True,
    help="dataset cache directory",
)
def pull(dataset: str, variant: str | None, data_dir: Path) -> None:
    """Download DATASET into the local cache."""
    path = get_loader(dataset, data_dir).pull(variant)
    click.echo(f"pulled {dataset} -> {path}")
