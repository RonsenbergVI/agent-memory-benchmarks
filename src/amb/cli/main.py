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

from amb import logs as logging
from amb.cli.commands import COMMANDS


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="amb", prog_name="amb")
@click.option(
    "--log-level",
    type=click.Choice(logging.LEVELS),
    default="warning",
    show_default=True,
    help="stderr log level; `debug` traces every question's retrieval",
)
def cli(log_level: str) -> None:
    """Agent memory benchmark harness."""
    logging.configure(log_level)


# attached here so a command never imports the group it hangs off
for command in COMMANDS:
    cli.add_command(command)


main = cli

if __name__ == "__main__":
    cli()
