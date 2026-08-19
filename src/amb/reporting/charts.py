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
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from amb.constants import LIGHT, DARK
from amb.contracts import Point, Series
from amb.reporting.helpers import (
    is_ratio,
    pretty,
    flatten,
    available_metrics,
    collect_series,
    collect_points,
    hues,
    round_bars,
    styled_axes,
    human_ticks,
    place_labels,
    pad_limits,
    pad_axis
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
    """Render the trade-off scatter and write it to `output`.

    Each system keeps the hue it wears on every other chart (sorted-name
    assignment); identity still rests on the direct labels, the hue is
    reinforcement. `better` is an optional corner cue naming the direction
    a system should move — mathtext arrows (r"$\\uparrow$ better"), since
    unicode arrows are missing from common fonts; a `\\downarrow` in it
    moves the cue to the bottom-right corner, clear of the origin cluster.
    `baseline` draws a horizontal reference line — the floor to beat; the
    caller explains it in the subtitle, since a text label on the line
    would collide with the dot labels. Returns the path written. Raises
    nothing on empty input — the caller checks, so it can report which
    metrics were available.
    """
    fig, ax, c = styled_axes(dark)

    hues = hues([p.label for p in points], c)
    colors = [hues[p.label] for p in points]
    xs, ys = [p.x for p in points], [p.y for p in points]
    # a soft halo under each dot, then the dot itself: >=8px, its system's
    # hue, 2px surface ring so overlaps stay legible
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

    # every dot is directly labelled: identity never rests on colour
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

    A system run at a single k appears as a lone dot; its line grows as
    sweeps accumulate. Hues follow the sorted-name series order, so a
    system keeps its colour from chart to chart. Returns the path written;
    raises nothing on empty input — the caller checks.

    Raises:
        ValueError: past 8 series the palette is exhausted; filter or
            facet instead of inventing a 9th hue.
    """
    if len(series) > len(LIGHT["categories"]):
        raise ValueError(
            f"{len(series)} systems exceed the {len(LIGHT['categories'])}-hue "
            "categorical palette; filter or facet instead"
        )
    fig, ax, c = _styled_axes(dark)

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
        # x: the k values actually run, as discrete ticks, with room on
        # the right for line-end labels
        ks = sorted({x for line in series for x in line.xs})
        span = (ks[-1] - ks[0]) or 1.0
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - span * 0.08, ks[-1] + span * 0.3)
        pad_axis(
            [y for line in series for y in line.ys],
            ax.set_ylim,
            ratio=is_ratio(y_label),
        )

        # the legend is identity's baseline; swept systems (true lines)
        # also get a direct end label while there are few enough to read
        legend = ax.legend(loc="best", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(c["muted"])
        swept = [line for line in series if len(line.xs) > 1]
        if 0 < len(swept) <= 4:
            place_labels(
                ax,
                [
                    Point(f"{line.label}  {line.ys[-1]:.2f}", line.xs[-1], line.ys[-1])
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

    Magnitude across a handful of named systems is a bar's job; identity
    sits on the axis, so every bar wears the single series hue and its
    value in ink at its end. Returns the path written; raises nothing on
    empty input — the caller checks.
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
        # headline in ink, context line in muted underneath it
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
