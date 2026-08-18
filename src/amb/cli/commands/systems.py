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

import click

from amb.base import discover_benchmarks


@click.command()
def systems() -> None:
    """List discovered memory system plugins.

    Raises:
        click.ClickException: if no integration is installed.
    """
    specs = discover_benchmarks()
    if not specs:
        raise click.ClickException(
            "no memory systems found — is the package installed?"
        )
    width = max(len(name) for name in specs)
    for name, spec in sorted(specs.items()):
        try:
            description = spec.describe()
        except Exception as exc:  # missing optional deps shouldn't break listing
            description = f"(unavailable: {type(exc).__name__})"
        click.echo(f"{name:<{width}}  {description}")
