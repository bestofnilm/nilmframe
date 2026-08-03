"""Generate the nilmframe mark: a wave curling into itself.

The mark is a single ribbon following a spiral whose radius collapses -- it enters
the frame as a broad open sweep and winds down to a point. A wave that breaks.

Two things make it look drawn rather than plotted, and both are why this is
generated instead of stroked in an editor:

**The width is a profile, not a setting.** The ribbon is a filled outline built by
offsetting a centreline by half its *local* width, so it can be broad where the
wave is heavy and vanish where it winds in. A constant stroke always reads as a
constant stroke.

**The mark is fitted to its own ink.** A spiral drawn in a 256 box does not fill
it, and a logo that floats off-centre in its own file lands off-centre everywhere
it is placed. The outline is measured and the whole path scaled into the frame, so
the optical weight sits in the middle by construction.

The gradient runs cyan through blue and violet to magenta. It carries no neutral,
which is what lets one file serve both the light and the dark page -- there is no
ink to invert.

Usage::

    python docs/_scripts/make_logo.py docs/_static
"""

from __future__ import annotations

import math
import pathlib
import sys

SIZE = 256
PADDING = 14

#: Cyan to magenta, no neutral -- so the same file reads on white and on near-black.
AURORA = [(0.0, "#22D3EE"), (0.32, "#3B82F6"), (0.64, "#A855F7"), (1.0, "#EC4899")]

# The shape. Tuned by rendering, not by reasoning: broad enough to hold up at the
# 44 px the sidebar draws it at, open enough to still read as a wave.
TURNS = 1.75
OUTER_RADIUS = 112.0
WIDTH = 70.0
FALLOFF = 1.30
HEAD = 0.32
START_ANGLE = 170.0
SAMPLES = 200


def _centreline(n: int = SAMPLES) -> list[tuple[float, float]]:
    """A spiral whose radius collapses from ``OUTER_RADIUS`` to a point."""
    a0 = math.radians(START_ANGLE)
    points = []
    for i in range(n):
        t = i / (n - 1)
        theta = a0 + TURNS * 2 * math.pi * t
        radius = 5 + (OUTER_RADIUS - 5) * (1 - t) ** FALLOFF
        points.append((128 + radius * math.cos(theta), 128 + radius * math.sin(theta)))
    return points


def _widths(n: int = SAMPLES) -> list[float]:
    """Broad at the crest, tapering to nothing at the centre.

    ``HEAD`` softens the outer end so the ribbon opens into the frame instead of
    starting on a blunt edge.
    """
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(WIDTH * min(1.0, t / HEAD) ** 0.7 * max(0.0, 1.0 - t) ** 0.9)
    return out


def _offsets(points, widths):
    """The two sides of the ribbon, each half a local width off the centreline."""
    n = len(points)
    left, right = [], []
    for i, (x, y) in enumerate(points):
        before, after = max(0, i - 1), min(n - 1, i + 1)
        dx = points[after][0] - points[before][0]
        dy = points[after][1] - points[before][1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        half = widths[i] / 2
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    return left, right


def _through(points) -> str:
    """Cubic beziers through the points -- Catmull-Rom, converted."""
    d = ""
    for i in range(len(points) - 1):
        p0 = points[max(0, i - 1)]
        p1, p2 = points[i], points[i + 1]
        p3 = points[min(len(points) - 1, i + 2)]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d += "C %.2f %.2f %.2f %.2f %.2f %.2f " % (*c1, *c2, *p2)
    return d


def path() -> str:
    """The closed ribbon outline, fitted to the frame."""
    left, right = _offsets(_centreline(), _widths())

    every = left + right
    x0, x1 = min(p[0] for p in every), max(p[0] for p in every)
    y0, y1 = min(p[1] for p in every), max(p[1] for p in every)
    span = SIZE - 2 * PADDING
    scale = min(span / (x1 - x0), span / (y1 - y0))
    ox = (SIZE - scale * (x1 - x0)) / 2 - scale * x0
    oy = (SIZE - scale * (y1 - y0)) / 2 - scale * y0

    def place(p):
        return (p[0] * scale + ox, p[1] * scale + oy)

    left = [place(p) for p in left]
    right = [place(p) for p in right]

    d = "M %.2f %.2f " % left[0] + _through(left)
    d += "L %.2f %.2f " % right[-1] + _through(list(reversed(right)))
    return d + "Z"


def svg(stops=AURORA) -> str:
    ramp = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in stops)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'width="{SIZE}" height="{SIZE}" fill="none" role="img" aria-label="nilmframe">'
        f'<defs><linearGradient id="nf-aurora" gradientUnits="userSpaceOnUse" '
        f'x1="18" y1="24" x2="238" y2="232">{ramp}</linearGradient></defs>'
        f'<path d="{path()}" fill="url(#nf-aurora)"/></svg>\n'
    )


def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    mark = svg()
    (out / "logo.svg").write_text(mark)
    (out / "favicon.svg").write_text(mark)
    print(f"wrote logo.svg and favicon.svg ({len(mark)} bytes each) to {out}")


if __name__ == "__main__":
    main()
