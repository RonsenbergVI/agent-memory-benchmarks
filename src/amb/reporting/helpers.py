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

from amb.contracts import Point, Series


def is_ratio(metric: str) -> bool:
    """Whether a metric name denotes a 0..1 ratio.

    Ratio axes are pinned to the full 0–1 band so a chart never implies
    scores above 100%; anything else (seconds, tokens) hugs its data —
    pinning latency to 1.0 would crush every dot into a corner.
    """
    return any(
        hint in metric
        for hint in ("precision", "recall", "f1", "accuracy", "exact", "judge")
    )


def pretty(name: str) -> str:
    """Human-readable form of a metric path, for labels and titles."""
    text = name.replace(".", " ").replace("_", " ")
    text = text.replace("f1", "F1")
    return re.sub(r"\bp(\d+) s\b", r"p\1 (s)", text)


def flatten(summary: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a run summary into dotted metric paths with numeric values.

    ``{"ingest": {"total_s": 4.0}}`` becomes ``{"ingest.total_s": 4.0}``, so
    nested sections are addressable as ``--x ingest.total_s``.
    """
    flat: dict[str, float] = {}
    for key, value in summary.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        elif isinstance(value, bool):
            continue  # a flag is not a measurement
        elif isinstance(value, int | float):
            flat[path] = float(value)
    return flat


def available_metrics(summaries: list[dict]) -> list[str]:
    """Every numeric metric path present in any of the summaries."""
    paths: set[str] = set()
    for summary in summaries:
        paths.update(flatten(summary))
    return sorted(paths)


def collect_series(summaries: list[dict], metric: str) -> list[Series]:
    """Build one line per system: `metric` at each k, newest run per (system, k).

    Systems whose runs never report the metric are left out; series come
    back in sorted-name order, each with its points in k order.
    """
    newest: dict[tuple[str, float], dict] = {}
    for summary in summaries:
        flat = flatten(summary)
        if metric not in flat or "k" not in flat:
            continue
        key = (str(summary.get("system", "?")), flat["k"])
        current = newest.get(key)
        if current is None or str(summary.get("run_id", "")) > str(
            current.get("run_id", "")
        ):
            newest[key] = summary
    series: dict[str, Series] = {}
    for (system, k), summary in sorted(newest.items()):
        line = series.setdefault(system, Series(label=system))
        line.xs.append(k)
        line.ys.append(flatten(summary)[metric])
    return [series[system] for system in sorted(series)]


def collect_points(summaries: list[dict], x: str, y: str) -> list[Point]:
    """Build one labelled point per system, newest run wins.

    A report directory usually holds several runs per system; the chart is
    "one dot per framework", so only the newest run of each is plotted.
    """
    newest: dict[str, dict] = {}
    for summary in summaries:
        flat = flatten(summary)
        if x not in flat or y not in flat:
            continue
        system = str(summary.get("system", "?"))
        current = newest.get(system)
        if current is None or str(summary.get("run_id", "")) > str(
            current.get("run_id", "")
        ):
            newest[system] = summary
    return [
        Point(label=system, x=flatten(s)[x], y=flatten(s)[y])
        for system, s in sorted(newest.items())
    ]


def hues(labels: list[str], c: dict) -> dict[str, str]:
    """Each label's hue, assigned in sorted-name order — the same rule the
    lines form uses, so a system keeps its colour from chart to chart."""
    palette = c["categories"]
    return {
        label: palette[i % len(palette)]
        for i, label in enumerate(sorted(set(labels)))
    }


def round_bars(fig: Any, ax: Any) -> None:
    """Soften every bar's corners with a pixel-true ~5px radius.

    The limits must be final before this runs: the radius is converted
    from pixels into each axis's data units through the live transform.
    """
    from matplotlib.patches import FancyBboxPatch

    fig.canvas.draw()  # the transform is only trustworthy after a draw
    inverse = ax.transData.inverted()
    (x0, y0), (x1, y1) = inverse.transform([(0, 0), (12, 12)])
    rx, ry = abs(x1 - x0), abs(y1 - y0)
    for rect in list(ax.patches):
        width = rect.get_width()
        if width <= 2 * rx:
            continue  # too short to round without eating the bar
        ax.add_patch(
            FancyBboxPatch(
                (rect.get_x(), rect.get_y()),
                width,
                rect.get_height(),
                boxstyle=f"round,pad=0,rounding_size={rx}",
                mutation_aspect=ry / rx,
                facecolor=rect.get_facecolor(),
                edgecolor="none",
                zorder=3,
            )
        )
        rect.remove()


def styled_axes(dark: bool) -> tuple[Any, Any, dict]:
    """One figure and axes on the chart surface, grid-only and recessive."""
    import matplotlib

    matplotlib.use("Agg")  # headless: never needs a display
    # crisper text where the system has it; CI falls back to DejaVu
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Helvetica Neue",
        "Arial",
        "DejaVu Sans",
    ]
    import matplotlib.pyplot as plt

    c = DARK if dark else LIGHT
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    # gridlines only — hairline, solid, behind the marks; no spines at all,
    # so the data floats on the surface instead of sitting in a box
    ax.set_axisbelow(True)
    ax.grid(True, color=c["grid"], linewidth=1.0, linestyle="-")
    for side in ax.spines.values():
        side.set_visible(False)
    return fig, ax, c


def human_ticks(value: float, _pos: int) -> str:
    """Tick text for large counts: 250k, 1.2M — never offset notation."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:g}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g}k"
    return f"{value:g}"


def place_labels(ax: Any, points: list[Point], color: str) -> None:
    """Direct-label every dot, nudging labels that would collide.

    Tries a few offsets per label and keeps the first that clears the ones
    already placed, so overlapping systems stay readable instead of printing
    on top of each other.
    """
    candidates = ((10, 4), (10, -12), (-10, 4), (-10, -12), (0, 12), (0, -16))
    placed: list[Any] = []
    renderer = ax.figure.canvas.get_renderer()
    for p in points:
        for dx, dy in candidates:
            text = ax.annotate(
                p.label,
                (p.x, p.y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="right" if dx < 0 else ("center" if dx == 0 else "left"),
                color=color,
                fontsize=9,
                zorder=4,
            )
            box = text.get_window_extent(renderer=renderer).expanded(1.05, 1.3)
            if not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            text.remove()
        else:  # every candidate collided; keep the default rather than drop it
            text = ax.annotate(
                p.label,
                (p.x, p.y),
                textcoords="offset points",
                xytext=(10, 4),
                color=color,
                fontsize=9,
                zorder=4,
            )
            placed.append(text.get_window_extent(renderer=renderer))


def pad_limits(ax: Any, points: list[Point], x_label: str, y_label: str) -> None:
    """Center the view on the dots, with room for their labels.

    A trade-off scatter's job is separating systems, so the spread between
    them gets the pixels — the axes hug the data instead of pinning score
    axes to the whole 0–1 band (which flattens a 0.42–0.48 field into one
    thin stripe). `_hug_axis` still clamps ratio axes to the honest range.
    """
    if not points:
        return
    _hug_axis([p.x for p in points], ax.set_xlim, ratio=is_ratio(x_label))
    _hug_axis([p.y for p in points], ax.set_ylim, ratio=is_ratio(y_label))


def _hug_axis(values: list[float], set_lim: Callable, ratio: bool = False) -> None:
    """Fit one axis tightly around its values, padded for marks and labels.

    Ratio axes are clamped so zooming never implies scores below 0 or
    above 1; inside those bounds the data decides the view.
    """
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    lower = lo - span * (0.08 if lo >= 0 else 0.15)
    upper = hi + span * 0.25  # head-room for the direct labels
    if ratio:
        lower, upper = max(lower, -0.02), min(upper, 1.02)
    set_lim(lower, upper)


def pad_axis(values: list[float], set_lim: Callable, ratio: bool = False) -> None:
    """Pad one axis's limits around its values.

    A ratio axis is pinned to the whole 0–1 band, so scores read against
    100%; any other axis hugs its data, with a little underhang so a
    zero-valued mark sits clear of the spine instead of clipped on it.
    """
    if ratio:
        set_lim(-0.05, 1.05)
        return
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    set_lim(
        lo - span * (0.06 if lo >= 0 else 0.15),
        hi + span * 0.25,
    )
