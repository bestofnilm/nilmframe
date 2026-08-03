"""Generate the nilmframe mark: a glowing bolt on indigo glass.

A macOS app-icon build — a squircle tile with depth, not a flat glyph. The bolt is
the only warm thing in the frame, and everything else in the file exists to make
that reading work.

Three things took the longest to get right:

**Warm on warm is mud.** The first attempt put amber and orange lobes behind a
yellow bolt and the whole tile turned brown, because a warm shape on a warm ground
has nothing to separate it. The glass is indigo — cool, so the bolt separates from
it — and the only warmth is the halo the bolt itself throws.

**Cool does not have to mean black.** The version after that pushed the glass to
near-black, which separated the bolt but crushed the corners into dead holes. The
floor of ``GLASS`` is a lit indigo, and ``VIGNETTE`` is kept low: the tile should
darken toward its edges, not fall out of the page.

**The bloom goes under the bolt, not over it.** Drawn on top it fogs the shape;
drawn underneath, the bolt sits inside its own light, which is what glowing looks
like.

**The hot core is a second, inset copy.** The bolt's outline scaled toward its own
centroid and blurred — so the middle reads hotter than the edges the way real
emission does, without a second gradient to keep in sync.

Corners are rounded by stroking the polygon in its own fill with a round join,
rather than by hand-fitting arcs into the path.

The squircle is a superellipse — Apple's continuous corner, not a rounded rect.

Usage::

    python docs/_scripts/make_logo.py docs/_static
"""

from __future__ import annotations

import math
import pathlib
import sys

SIZE = 512

#: Lit indigo, not near-black: cool enough to separate the warm bolt, bright
#: enough that no corner of the tile reads as a hole.
GLASS = ("#241C56", "#1B1545", "#131034")

#: Light sources behind the glass, as ``(x, y, radius, colour, opacity)``. Only the
#: first is warm — it is the bolt's own spill. The rest give the tile depth, and
#: between them they cover every corner, which is what stops one going dead.
LOBES = [
    (256, 250, 230, "#F59E0B", 0.30),
    (85, 80, 240, "#6366F1", 0.60),
    (505, 175, 240, "#8B5CF6", 0.55),
    (430, 500, 240, "#3B82F6", 0.50),
    (60, 470, 230, "#7C3AED", 0.45),
    (500, 480, 200, "#A855F7", 0.35),
]

#: How hard the tile darkens toward its edges. Low on purpose — this is the knob
#: that turned the corners black.
VIGNETTE = 0.16

BOLT_HEIGHT = 340.0
BOLT_WIDTH = 150.0
CORNER = 16.0        #: stroke width that rounds the polygon's corners
GLOW = 54.0          #: bloom radius under the bolt
CORE = 0.72          #: how far the hot core is inset, as a fraction of the outline


def squircle(inset: float = 0.0, n: float = 4.6, steps: int = 200) -> str:
    """A superellipse — the continuous corner, not a rounded rectangle."""
    a, o = SIZE / 2 - inset, SIZE / 2
    points = []
    for k in range(steps):
        t = 2 * math.pi * k / steps
        c, s = math.cos(t), math.sin(t)
        points.append((
            o + a * math.copysign(abs(c) ** (2 / n), c),
            o + a * math.copysign(abs(s) ** (2 / n), s),
        ))
    return "M %.2f %.2f " % points[0] + " ".join("L %.2f %.2f" % p for p in points[1:]) + " Z"


def bolt_points(cx=256.0, cy=256.0, height=BOLT_HEIGHT, width=BOLT_WIDTH,
                waist=0.30, kick=0.30):
    """Six points. The lower half repeats the upper half turned about the centre,
    which is what stops a bolt reading as a broken arrow."""
    top, bottom = cy - height / 2, cy + height / 2
    mid = cy + height * (waist - 0.5)
    return [
        (cx + width * 0.42, top),
        (cx - width * 0.50, mid + height * 0.10),
        (cx - width * 0.04, mid + height * 0.10),
        (cx - width * 0.42, bottom),
        (cx + width * 0.52, cy - height * kick * 0.34),
        (cx + width * 0.05, cy - height * kick * 0.34),
    ]


def _poly(points) -> str:
    return "M " + " L ".join("%.1f %.1f" % p for p in points) + " Z"


def _inset(points, factor):
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [(cx + (x - cx) * factor, cy + (y - cy) * factor) for x, y in points]


def svg() -> str:
    points = bolt_points()
    outline = _poly(points)
    core = _poly(_inset(points, CORE))
    lobes = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" opacity="{o}"/>'
        for x, y, r, c, o in LOBES
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}" fill="none" role="img" aria-label="nilmframe">
  <defs>
    <clipPath id="nf-tile"><path d="{squircle()}"/></clipPath>
    <linearGradient id="nf-glass" x1="0.1" y1="0" x2="0.9" y2="1">
      <stop offset="0" stop-color="{GLASS[0]}"/>
      <stop offset="0.55" stop-color="{GLASS[1]}"/>
      <stop offset="1" stop-color="{GLASS[2]}"/>
    </linearGradient>
    <linearGradient id="nf-bolt" x1="0.15" y1="0" x2="0.85" y2="1">
      <stop offset="0" stop-color="#FFFBEB"/>
      <stop offset="0.26" stop-color="#FDE047"/>
      <stop offset="0.62" stop-color="#FBBF24"/>
      <stop offset="1" stop-color="#F97316"/>
    </linearGradient>
    <radialGradient id="nf-spec">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.20"/>
      <stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="nf-sheen" x1="0" y1="0" x2="0.22" y2="1">
      <stop offset="0" stop-color="#FFFFFF" stop-opacity="0.16"/>
      <stop offset="0.42" stop-color="#FFFFFF" stop-opacity="0.03"/>
      <stop offset="1" stop-color="#FFFFFF" stop-opacity="0"/>
    </linearGradient>
    <radialGradient id="nf-vig" cx="0.5" cy="0.44" r="0.78">
      <stop offset="0.58" stop-color="#000000" stop-opacity="0"/>
      <stop offset="1" stop-color="#000000" stop-opacity="{VIGNETTE}"/>
    </radialGradient>
    <filter id="nf-soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="50"/>
    </filter>
    <filter id="nf-bloom" x="-70%" y="-70%" width="240%" height="240%">
      <feGaussianBlur stdDeviation="{GLOW}"/>
    </filter>
    <filter id="nf-core" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="11"/>
    </filter>
  </defs>
  <g clip-path="url(#nf-tile)">
    <rect width="{SIZE}" height="{SIZE}" fill="url(#nf-glass)"/>
    <g filter="url(#nf-soft)">{lobes}</g>
    <rect width="{SIZE}" height="{SIZE}" fill="url(#nf-vig)"/>
    <ellipse cx="150" cy="112" rx="200" ry="130" fill="url(#nf-spec)" transform="rotate(-24 150 112)"/>
    <rect width="{SIZE}" height="{SIZE}" fill="url(#nf-sheen)"/>
    <path d="{outline}" fill="#FDE047" filter="url(#nf-bloom)" opacity="0.9"/>
    <path d="{outline}" fill="url(#nf-bolt)" stroke="url(#nf-bolt)" stroke-width="{CORNER}"
          stroke-linejoin="round" stroke-linecap="round"/>
    <path d="{core}" fill="#FFFEF7" filter="url(#nf-core)" opacity="0.55"/>
    <path d="{outline}" fill="none" stroke="#FFFDF2" stroke-width="3" stroke-linejoin="round"
          opacity="0.55"/>
  </g>
  <path d="{squircle(inset=1.8)}" fill="none" stroke="#FFFFFF" stroke-opacity="0.24" stroke-width="3.4"/>
</svg>
"""


def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    mark = svg()
    (out / "logo.svg").write_text(mark)
    (out / "favicon.svg").write_text(mark)
    print(f"wrote logo.svg and favicon.svg ({len(mark)} bytes each) to {out}")


if __name__ == "__main__":
    main()
