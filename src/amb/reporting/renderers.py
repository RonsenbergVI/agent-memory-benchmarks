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

from collections.abc import Callable, Sequence

from amb.base.reporting import Renderer
from amb.contracts import Block, Figure, Heading, Paragraph, Rule, Table
from amb.reporting.run import markdown_table


class MarkdownRenderer(Renderer):
    """Renders blocks as GitHub-flavoured markdown."""

    def render(self, blocks: Sequence[Block]) -> str:
        """Return the markdown for `blocks`, one blank line between each."""
        return "\n\n".join(self.block(block) for block in blocks) + "\n"

    def block(self, block: Block) -> str:
        """Return one block's markdown.

        Raises:
            TypeError: on a block type this renderer does not know.
        """
        match block:
            case Heading():
                return f"{'#' * block.level} {block.text}"
            case Paragraph():
                return block.text
            case Table():
                return markdown_table(list(block.header), [list(r) for r in block.rows])
            case Figure():
                return f"![{block.alt}]({block.path.as_posix()})"
            case Rule():
                return "---"
            case _:
                raise TypeError(f"no markdown for {type(block).__name__}")


# output formats by name; a new format registers here, no section changes
RENDERERS: dict[str, Callable[[], Renderer]] = {"markdown": MarkdownRenderer}
FORMATS = tuple(RENDERERS)


def get_renderer(fmt: str) -> Renderer:
    """The renderer registered under `fmt`, freshly instantiated.

    Raises:
        ValueError: on a format no renderer is registered for.
    """
    try:
        return RENDERERS[fmt]()
    except KeyError as exc:
        raise ValueError(
            f"unknown format {fmt!r}; known: {', '.join(RENDERERS)}"
        ) from exc
