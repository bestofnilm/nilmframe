"""Generate the nilmframe mark: a leaf whose midrib is a signal.

Green because the subject is household energy. A waveform for the midrib because
the subject is *reading* that energy -- the mark says both things at once without
being a diagram of either.

Two pieces of construction are worth knowing before changing anything:

**A leaf is a ribbon.** It is a curved spine with a lens-shaped width profile, so
it is built the same way any other ribbon here is: sample the spine, offset each
sample by half the local width along the normal, and close the two sides into one
filled outline. Changing ``LEAF_WIDTH`` or ``BEND`` reshapes the blade without
touching a single path coordinate.

**The midrib is a hole, not a stroke.** It is cut with a mask rather than drawn in
white, so the page shows through it: light on the light theme, dark on the dark
one, correct on both without a second file. Same for the side veins, if enabled.

The blade is fitted to its own bounding box before it is written, so the mark
fills the frame it ships in and lands centred wherever it is placed.

Usage::

    python docs/_scripts/make_logo.py docs/_static
"""

from __future__ import annotations

import math
import pathlib
import sys

SIZE = 256
PADDING = 16
SAMPLES = 64

#: Fresh at the tip, deep at the base. No neutral, so one file serves both themes.
GREENS = [(0.0, "#BEF264"), (0.4, "#4ADE80"), (1.0, "#15803D")]

BASE = (58.0, 214.0)     #: where the stem meets the blade
TIP = (206.0, 44.0)      #: the point
LEAF_WIDTH = 132.0       #: widest span across the blade
BEND = 32.0              #: how far the spine bows out of the straight line
RIB_AMPLITUDE = 24.0     #: how far the midrib swings off the spine
RIB_CYCLES = 1.6         #: how many swings it makes along the blade
RIB_STROKE = 12.0        #: the cut's width
VEIN_PAIRS = 0           #: side veins per side; they clutter the 20 px favicon


def _quad(p0, p1, p2, t):
    a, b, c = (1 - t) ** 2, 2 * (1 - t) * t, t * t
    return (a * p0[0] + b * p1[0] + c * p2[0], a * p0[1] + b * p1[1] + c * p2[1])


def _spine(base, tip, bend, n=SAMPLES):
    """A gently bowed line from base to tip -- the blade's centre."""
    mx, my = (base[0] + tip[0]) / 2, (base[1] + tip[1]) / 2
    dx, dy = tip[0] - base[0], tip[1] - base[1]
    length = math.hypot(dx, dy) or 1.0
    control = (mx - dy / length * bend, my + dx / length * bend)
    return [_quad(base, control, tip, i / (n - 1)) for i in range(n)]


def _widths(width, n=SAMPLES, skew=0.86, fullness=0.72):
    """Zero at both ends, fullest just past the base.

    ``skew`` below 1 moves the widest point toward the stem, which is where it is
    on a real leaf; ``fullness`` below 1 broadens the whole blade rather than
    leaving it a pointed almond.
    """
    return [
        width * max(0.0, math.sin(math.pi * (i / (n - 1)) ** skew)) ** fullness
        for i in range(n)
    ]


def _normals(points):
    """Unit normal at each sample, from the direction of its neighbours."""
    n = len(points)
    out = []
    for i in range(n):
        before, after = max(0, i - 1), min(n - 1, i + 1)
        dx = points[after][0] - points[before][0]
        dy = points[after][1] - points[before][1]
        length = math.hypot(dx, dy) or 1.0
        out.append((-dy / length, dx / length))
    return out


def _sides(points, widths):
    left, right = [], []
    for (x, y), (nx, ny), w in zip(points, _normals(points), widths, strict=True):
        left.append((x + nx * w / 2, y + ny * w / 2))
        right.append((x - nx * w / 2, y - ny * w / 2))
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
        d += "C %.1f %.1f %.1f %.1f %.1f %.1f " % (*c1, *c2, *p2)
    return d


def _fitter(points, widths):
    """A transform putting the blade's own bounding box in the middle of the frame."""
    left, right = _sides(points, widths)
    every = left + right
    x0, x1 = min(p[0] for p in every), max(p[0] for p in every)
    y0, y1 = min(p[1] for p in every), max(p[1] for p in every)
    span = SIZE - 2 * PADDING
    scale = min(span / (x1 - x0), span / (y1 - y0))
    ox = (SIZE - scale * (x1 - x0)) / 2 - scale * x0
    oy = (SIZE - scale * (y1 - y0)) / 2 - scale * y0
    return lambda p: (p[0] * scale + ox, p[1] * scale + oy)


def _blade(points, widths, place) -> str:
    left, right = _sides(points, widths)
    left = [place(p) for p in left]
    right = [place(p) for p in right]
    d = "M %.1f %.1f " % left[0] + _through(left)
    d += "L %.1f %.1f " % right[-1] + _through(list(reversed(right)))
    return d + "Z"


def _line(points, place) -> str:
    placed = [place(p) for p in points]
    return "M %.1f %.1f " % placed[0] + _through(placed)


def _wavy(points, amplitude, cycles):
    """The spine displaced sideways by a sine, damped to nothing at both ends.

    Undamped, the wave would break out through the tip, where the blade has no
    width left to contain it.
    """
    out = []
    for i, ((x, y), (nx, ny)) in enumerate(zip(points, _normals(points), strict=True)):
        t = i / (len(points) - 1)
        k = amplitude * math.sin(2 * math.pi * cycles * t) * math.sin(math.pi * t)
        out.append((x + nx * k, y + ny * k))
    return out


def _veins(points, widths, pairs, reach=0.60, tilt=38.0):
    """Side veins that stay inside the blade and lean toward the tip."""
    out, n = [], len(points)
    normals = _normals(points)
    for k in range(pairs):
        i = int((0.24 + 0.52 * (k / max(1, pairs - 1))) * (n - 1))
        px, py = points[i]
        nx, ny = normals[i]
        tx, ty = -ny, nx
        length = widths[i] / 2 * reach
        angle = math.radians(tilt)
        for sign in (1, -1):
            ux, uy = nx * sign, ny * sign
            ex = ux * math.cos(angle) + tx * math.sin(angle)
            ey = uy * math.cos(angle) + ty * math.sin(angle)
            end = (px + ex * length, py + ey * length)
            control = (px + ux * length * 0.62, py + uy * length * 0.62)
            out.append([_quad((px, py), control, end, j / 11) for j in range(12)])
    return out


def svg() -> str:
    points = _spine(BASE, TIP, BEND)
    widths = _widths(LEAF_WIDTH)
    place = _fitter(points, widths)

    cuts = (
        f'<path d="{_line(_wavy(points, RIB_AMPLITUDE, RIB_CYCLES), place)}" '
        f'stroke="#000" stroke-width="{RIB_STROKE}" stroke-linecap="round" fill="none"/>'
    )
    for vein in _veins(points, widths, VEIN_PAIRS):
        cuts += (
            f'<path d="{_line(vein, place)}" stroke="#000" stroke-width="6" '
            f'stroke-linecap="round" fill="none"/>'
        )

    ramp = "".join(f'<stop offset="{o}" stop-color="{c}"/>' for o, c in GREENS)
    stem = _spine(BASE, (BASE[0] - 32, BASE[1] + 28), -13, 16)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" '
        f'width="{SIZE}" height="{SIZE}" fill="none" role="img" aria-label="nilmframe">'
        f'<defs>'
        f'<linearGradient id="nf-leaf" gradientUnits="userSpaceOnUse" x1="30" y1="20" '
        f'x2="226" y2="236">{ramp}</linearGradient>'
        f'<mask id="nf-rib"><rect width="{SIZE}" height="{SIZE}" fill="#fff"/>{cuts}</mask>'
        f'</defs>'
        f'<path d="{_blade(points, widths, place)}" fill="url(#nf-leaf)" mask="url(#nf-rib)"/>'
        f'<path d="{_line(stem, place)}" stroke="{GREENS[-1][1]}" stroke-width="10" '
        f'stroke-linecap="round" fill="none"/>'
        f'</svg>\n'
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
