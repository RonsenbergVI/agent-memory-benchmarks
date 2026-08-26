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

import math
from pathlib import Path
from typing import Any

from amb.constants import LIGHT
from amb.contracts import Point, Series
from amb.reporting.helpers import (
    hues,
    human_ticks,
    is_ratio,
    pad_axis,
    pad_limits,
    place_labels,
    pretty,
    round_bars,
    styled_axes,
)


def scatter(
    points: list[Point],
    x_label: str,
    y_label: str,
    output: Path,
    title: str | None = None,
    subtitle: str | None = None,
    better: str | None = None,
    baseline: float | None = None,
    dark: bool = False,
) -> Path:
    r"""Render the trade-off scatter and write it to `output`.

    `better` is a corner cue in mathtext arrows (unicode arrows are missing
    from common fonts); a `\\downarrow` moves it to the bottom-right, clear
    of the origin cluster. `baseline` draws the floor-to-beat line, explained
    in the subtitle since a line label would collide with the dot labels.
    Empty input is the caller's job to check. Returns the path written.
    """
    fig, ax, c = styled_axes(dark)

    hue_map = hues([p.label for p in points], c)
    colors = [hue_map[p.label] for p in points]
    xs, ys = [p.x for p in points], [p.y for p in points]
    # halo under each dot; 2px surface ring keeps overlaps legible
    ax.scatter(xs, ys, s=260, c=colors, alpha=0.15, linewidths=0, zorder=2)
    ax.scatter(
        xs,
        ys,
        s=110,
        c=colors,
        edgecolors=c["surface"],
        linewidths=2.0,
        zorder=3,
    )

    if baseline is not None:
        ax.axhline(baseline, color=c["muted"], linewidth=1.0, alpha=0.5, zorder=1)

    # every dot directly labelled: identity never rests on colour
    place_labels(ax, points, color=c["text"])
    pad_limits(ax, points, x_label, y_label)
    if better:
        x, y, ha, va = (
            (0.98, 0.03, "right", "bottom")
            if "down" in better
            else (0.02, 0.97, "left", "top")
        )
        ax.text(
            x,
            y,
            better,
            transform=ax.transAxes,
            color=c["muted"],
            fontsize=10,
            ha=ha,
            va=va,
        )
    return _finish(fig, ax, c, x_label, y_label, title, output, subtitle)


def lines(
    series: list[Series],
    x_label: str,
    y_label: str,
    output: Path,
    title: str | None = None,
    subtitle: str | None = None,
    dark: bool = False,
) -> Path:
    """Render one line per system over k and write it to `output`.

    A single-k system appears as a lone dot; hues follow sorted-name order so
    a system keeps its colour from chart to chart. Empty input is the caller's
    job to check. Returns the path written.

    Raises:
        ValueError: past 8 series the palette is exhausted; filter or facet.
    """
    if len(series) > len(LIGHT["categories"]):
        raise ValueError(
            f"{len(series)} systems exceed the {len(LIGHT['categories'])}-hue "
            "categorical palette; filter or facet instead"
        )
    fig, ax, c = styled_axes(dark)

    for line, color in zip(series, c["categories"], strict=False):
        ax.plot(
            line.xs,
            line.ys,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=8,
            markeredgecolor=c["surface"],
            markeredgewidth=2.0,
            zorder=3,
            label=line.label,
        )

    if series:
        # discrete ticks at the k values run, right room for line-end labels
        ks = sorted({x for line in series for x in line.xs})
        span = (ks[-1] - ks[0]) or 1.0
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - span * 0.08, ks[-1] + span * 0.3)
        pad_axis(
            [y for line in series for y in line.ys],
            ax.set_ylim,
            ratio=is_ratio(y_label),
        )

        # legend carries identity; swept lines also get a direct end label
        # while there are few enough to read
        legend = ax.legend(loc="best", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(c["muted"])
        swept = [line for line in series if len(line.xs) > 1]
        if 0 < len(swept) <= 4:
            place_labels(
                ax,
                [
                    Point(
                        label=f"{line.label}  {line.ys[-1]:.2f}",
                        x=line.xs[-1],
                        y=line.ys[-1],
                    )
                    for line in swept
                ],
                color=c["text"],
            )
    return _finish(fig, ax, c, x_label, y_label, title, output, subtitle)


def bars(
    items: list[tuple[str, float]],
    x_label: str,
    output: Path,
    title: str | None = None,
    subtitle: str | None = None,
    dark: bool = False,
) -> Path:
    """Render one metric as horizontal bars, one per system, largest on top.

    Identity sits on the axis, so every bar wears the single series hue with
    its value at its end. Empty input is the caller's job to check. Returns
    the path written.
    """
    fig, ax, c = styled_axes(dark)
    ax.grid(False, axis="y")  # gridlines only along the value axis

    ordered = sorted(items, key=lambda item: item[1])
    labels = [label for label, _ in ordered]
    values = [value for _, value in ordered]
    ax.barh(labels, values, height=0.55, color=c["series"], zorder=3)
    for i, value in enumerate(values):
        ax.annotate(
            f"{value:.3f}",
            (value, i),
            textcoords="offset points",
            xytext=(8, -3.5),
            color=c["text"],
            fontsize=9.5,
            fontweight="medium",
            zorder=4,
        )
    if values:
        pad_axis(values, ax.set_xlim, ratio=is_ratio(x_label))
        ax.set_xlim(left=0.0)  # bars are anchored at zero, so the axis is too
        round_bars(fig, ax)
    return _finish(fig, ax, c, x_label, "", title, output, subtitle)


def table(
    header: list[str],
    rows: list[list[str]],
    output: Path,
    title: str | None = None,
    subtitle: str | None = None,
    dark: bool = False,
) -> Path:
    """Render a small comparison table as a figure and write it to `output`.

    The README embeds this instead of markdown so every published number lives
    under `plots/`, where CI blocks hand edits. First two columns are identity
    (left-aligned); the rest are numbers (right-aligned). Empty input is the
    caller's job to check. Returns the path written.
    """
    fig, ax, c = styled_axes(dark)
    ax.grid(False)
    ax.set_axis_off()

    # column widths from content; the figure grows so text never squeezes
    cells = [header, *rows]
    widths = [max(len(str(row[j])) for row in cells) for j in range(len(header))]
    gutter = 3.0
    total = sum(widths) + gutter * (len(widths) - 1)
    edges: list[tuple[float, float]] = []
    cursor = 0.0
    for width in widths:
        edges.append((cursor / total, (cursor + width) / total))
        cursor += width + gutter
    fig.set_size_inches(max(6.4, total * 0.093), 0.44 * len(cells) + 0.72)

    def place(row: list[str], y: float, color: str, size: float, weight: str) -> None:
        for j, cell in enumerate(row):
            left, right = edges[j]
            identity = j < 2
            ax.text(
                left if identity else right,
                y,
                str(cell),
                transform=ax.transAxes,
                ha="left" if identity else "right",
                va="center",
                color=color,
                fontsize=size,
                fontweight=weight,
            )

    step = 1.0 / len(cells)
    place(header, 1.0 - step / 2, c["muted"], 9.0, "medium")
    for i, row in enumerate(rows):
        y = 1.0 - (i + 1.5) * step
        place(row, y, c["text"], 10.0, "normal")
        # hairline separator above each body row
        ax.axhline(
            y + step / 2,
            xmin=0.0,
            xmax=1.0,
            color=c["grid"],
            linewidth=1.0,
            zorder=1,
        )
    return _finish(fig, ax, c, "", "", title, output, subtitle)


def _finish(
    fig: Any,
    ax: Any,
    c: dict,
    x_label: str,
    y_label: str,
    title: str | None,
    output: Path,
    subtitle: str | None = None,
) -> Path:
    """Apply the text tokens, save the figure, and return the path written."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    ax.set_xlabel(pretty(x_label), color=c["muted"], fontsize=9)
    ax.set_ylabel(pretty(y_label), color=c["muted"], fontsize=9)
    if title:
        ax.set_title(
            title,
            color=c["text"],
            fontsize=13,
            fontweight="bold",
            loc="left",
            pad=24 if subtitle else 12,
        )
        if subtitle:
            ax.text(
                0.0,
                1.03,
                subtitle,
                transform=ax.transAxes,
                color=c["muted"],
                fontsize=9,
            )
    ax.tick_params(colors=c["muted"], labelsize=8, length=0)
    # token-scale axes read as 250k / 1.2M, not 0.25 against a "1e6" offset
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = axis.get_data_interval()
        if math.isfinite(hi) and max(abs(lo), abs(hi)) >= 10_000:
            axis.set_major_formatter(FuncFormatter(human_ticks))

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=c["surface"])
    plt.close(fig)
    return output
