"""Render the documentation's dataset figures, light and dark.

Run on a machine that has already fetched the subsets (see ``downloading.md``);
the figures land in ``~/nf-work/plots`` and are copied into
``docs/_static/datasets``.

**Every dataset figure has the same shape, and shows raw signals only.** Left, one
channel in detail: voltage above current for the waveform corpora, the aggregate
for the power-only ones. Right, three more channels of the same corpus, stacked.
Nothing here is a derived representation -- a V-I trajectory is a *feature*, and
it belongs in the chapter about extracting features rather than the one about
what a corpus contains. Mixing the two made the corpora look unlike each other
for reasons that had nothing to do with the corpora.

Waveform panels always show three whole mains cycles at the dataset's own line
frequency, so the horizontal axis means the same thing everywhere and no panel
clips a cycle in half.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

logging.disable(logging.WARNING)

CACHE = Path.home() / "nf-work/cache"
OUT = Path.home() / "nf-work/plots"
OUT.mkdir(parents=True, exist_ok=True)

#: Tall enough that three stacked panels each get room for whole cycles.
SIZE = (11.0, 6.4)

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e3e2df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333331",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

CYCLES = 3.0
SUMMARY: dict[str, dict] = {}


def style(theme: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "text.color": theme["ink"],
            "axes.labelcolor": theme["muted"],
            "axes.edgecolor": theme["grid"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "grid.color": theme["grid"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
        }
    )


def tidy(ax, theme: dict) -> None:
    ax.grid(True, alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def render(name: str, build, size=SIZE) -> None:
    for mode, theme in THEMES.items():
        style(theme)
        fig = plt.figure(figsize=size, dpi=155)
        try:
            build(fig, theme)
        except Exception as exc:
            plt.close(fig)
            print(f"  !! {name} ({mode}): {type(exc).__name__}: {exc}")
            return
        fig.savefig(OUT / f"{name}-{mode}.png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    print(f"  ok {name}")


def cycle_window(rec, f0: float, cycles: float = CYCLES):
    """Exactly ``cycles`` whole mains cycles from the middle of a recording."""
    span = max(2, min(round(rec.fs * cycles / f0), rec.n_samples))
    start = max(0, rec.n_samples // 2 - span // 2)
    stop = start + span
    return (np.arange(stop - start) / rec.fs) * 1000.0, start, stop


def grid_for(fig):
    """Six rows so the left column splits in two and the right in three."""
    return GridSpec(6, 2, figure=fig, hspace=1.15, wspace=0.26)


def detail_pair(fig, grid, theme, rec, title, f0):
    """Left column: voltage above current, one channel, whole cycles."""
    top = fig.add_subplot(grid[0:3, 0])
    bottom = fig.add_subplot(grid[3:6, 0], sharex=top)
    t, a, b = cycle_window(rec, f0)

    top.plot(t, rec.signals["v"][a:b], color=theme["series"][0])
    top.set_ylabel("volts")
    top.set_title(title, loc="left", color=theme["ink"])
    top.tick_params(labelbottom=False)
    tidy(top, theme)

    bottom.plot(t, rec.signals["i"][a:b], color=theme["series"][1])
    bottom.set_ylabel("amperes")
    bottom.set_xlabel("milliseconds")
    tidy(bottom, theme)
    return top, bottom


def channel_stack(fig, grid, theme, entries, heading, *, f0=None, hours=None):
    """Right column: three channels of the corpus, raw, stacked on one axis."""
    axes = []
    for n, ((label, rec), colour) in enumerate(zip(entries, theme["series"], strict=False)):
        ax = fig.add_subplot(grid[2 * n : 2 * n + 2, 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        if hours is not None:
            count = min(rec.n_samples, int(rec.fs * 3600 * hours))
            ax.plot(
                np.arange(count) / rec.fs / 3600.0,
                rec.signals["p"][:count],
                color=colour,
                linewidth=1.0,
            )
            ax.set_ylabel("watts")
        else:
            t, a, b = cycle_window(rec, f0)
            ax.plot(t, rec.signals["i"][a:b], color=colour)
            ax.set_ylabel("amperes")
        ax.set_title(label, loc="left", color=colour, fontsize=9)
        tidy(ax, theme)
        if n < len(entries) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("hours" if hours is not None else "milliseconds")
    if heading:
        axes[0].annotate(
            heading,
            xy=(0.0, 1.55),
            xycoords="axes fraction",
            color=theme["ink"],
            fontsize=9.5,
            fontweight="semibold",
        )
    return axes


def by_class(recs):
    out: dict[str, object] = {}
    for rec in recs:
        out.setdefault(rec.appliance or "empty grid", rec)
    return out


def strongest(recs, n=3):
    """Distinct appliance classes, largest current first."""
    seen, pick = set(), []
    for rec in sorted(recs, key=lambda r: -float(np.abs(r.signals["i"]).max())):
        if rec.appliance not in seen:
            seen.add(rec.appliance)
            pick.append(rec)
    return pick[:n]


# --------------------------------------------------------------------------- #


def do_blond():
    from nilmframe.readers import BLOND

    root = CACHE / "blond"
    recs = list(
        BLOND(
            root / "BLOND-50",
            appliance_log=root / "appliance_log.json",
            units=["clear", "medal-1"],
            max_files=1,
            max_seconds=1.0,
        )
    )
    mains = [r for r in recs if r.kind.value == "mains"]
    subs = [r for r in recs if r.kind.value == "submeter"]
    SUMMARY["blond"] = {
        "mains_rate_hz": mains[0].fs,
        "submeter_rate_hz": subs[0].fs,
        "phases": [r.meta["phase"] for r in mains],
        "appliances": sorted({r.appliance for r in subs}),
    }

    def build(fig, theme):
        grid = grid_for(fig)
        detail_pair(fig, grid, theme, mains[0], "mains phase L1, 50 kHz", 50.0)
        pick = strongest(subs)
        channel_stack(
            fig,
            grid,
            theme,
            [(r.appliance, r) for r in pick],
            "metered sockets on that phase, 6.4 kHz",
            f0=50.0,
        )

    render("blond", build)


def do_fired():
    from nilmframe.readers import FIRED

    root = CACHE / "fired"
    wave = list(FIRED(root, resolution="highFreq", max_seconds=1.0))
    power = list(FIRED(root, resolution="1Hz", max_seconds=86400))
    mains = [r for r in power if r.kind.value == "mains"]
    subs = [r for r in power if r.kind.value == "submeter"]
    SUMMARY["fired"] = {
        "waveform_rate_hz": wave[0].fs if wave else None,
        "summary_rate_hz": power[0].fs,
        "phases": len(mains),
        "appliances": sorted({r.appliance for r in subs}),
    }

    def build(fig, theme):
        grid = grid_for(fig)
        detail_pair(fig, grid, theme, wave[0], f"{wave[0].appliance}, 2 kHz", 50.0)
        order = [("aggregate (smart meter L1)", mains[0])]
        order += [
            (r.appliance, r) for r in sorted(subs, key=lambda r: -float(np.ptp(r.signals["p"])))[:2]
        ]
        channel_stack(fig, grid, theme, order, "the same apartment at 1 Hz", hours=24.0)

    render("fired", build)


def do_ukdale():
    from nilmframe.readers import UKDALE

    root = CACHE / "ukdale"
    meters: dict[str, object] = {}
    for rec in UKDALE(root / "low_freq", high_freq=False, houses=[1]):
        key = "aggregate (mains)" if rec.kind.value == "mains" else rec.appliance
        if key not in meters or rec.n_samples > meters[key].n_samples:
            meters[key] = rec
    wave = next(
        r
        for r in UKDALE(
            root / "low_freq",
            high_freq_root=root / "high_freq",
            houses=[1],
            max_hf_files=1,
            max_seconds=1.0,
        )
        if r.fs > 1000
    )
    SUMMARY["ukdale"] = {
        "lf_rate_hz": round(next(iter(meters.values())).fs, 4),
        "hf_rate_hz": wave.fs,
        "channels": sorted(meters),
    }

    def build(fig, theme):
        grid = grid_for(fig)
        detail_pair(fig, grid, theme, wave, "house 1 mains, 16 kHz", 50.0)
        others = sorted(
            (k for k in meters if k != "aggregate (mains)"),
            key=lambda k: -float(np.ptp(meters[k].signals["p"][: int(meters[k].fs * 3600 * 6)])),
        )
        order = ["aggregate (mains)", *others[:2]]
        channel_stack(
            fig,
            grid,
            theme,
            [(k, meters[k]) for k in order],
            "the same house at 1/6 Hz",
            hours=6.0,
        )

    render("ukdale", build)


def do_smartnialm():
    from nilmframe.readers import SmartNIALM

    runs: dict[str, list] = {}
    for rec in SmartNIALM(CACHE / "snm", buildings=[1]):
        key = "aggregate (smart meter)" if rec.kind.value == "mains" else rec.appliance
        runs.setdefault(key, []).append(rec)

    def slice_at(key, start, hours):
        for rec in runs.get(key, []):
            begin = rec.t0
            end = rec.t0 + rec.n_samples / rec.fs
            if begin <= start < end:
                first = int((start - begin) * rec.fs)
                count = min(int(hours * 3600 * rec.fs), rec.n_samples - first)
                if count > 10:
                    return (
                        np.arange(count) / rec.fs / 3600.0,
                        rec.signals["p"][first : first + count],
                    )
        return None

    longest = max(runs["aggregate (smart meter)"], key=lambda r: r.n_samples)
    per_day = int(24 * 3600 * longest.fs)
    blocks = max(1, longest.n_samples // per_day)
    best = max(
        (float(np.std(longest.signals["p"][n * per_day : (n + 1) * per_day])), n)
        for n in range(blocks)
    )
    start = longest.t0 + best[1] * per_day / longest.fs

    def activity(key):
        got = slice_at(key, start, 24.0)
        return float(np.ptp(got[1])) if got is not None else -1.0

    order = ["aggregate (smart meter)"]
    order += sorted((k for k in runs if k != order[0]), key=activity, reverse=True)[:2]
    SUMMARY["smartnialm"] = {
        "rate_hz": longest.fs,
        "channels": sorted(runs),
        "runs_per_channel": {k: len(v) for k, v in sorted(runs.items())},
        "longest_run_days": round(longest.n_samples / longest.fs / 86400, 1),
    }

    def build(fig, theme):
        grid = grid_for(fig)
        # No waveform exists here, so the detail panel is the aggregate itself.
        ax = fig.add_subplot(grid[:, 0])
        got = slice_at(order[0], start, 24.0)
        ax.plot(got[0], got[1], color=theme["series"][0], linewidth=0.9)
        ax.set_title("building 1 aggregate, one day at 0.2 Hz", loc="left", color=theme["ink"])
        ax.set_xlabel("hours")
        ax.set_ylabel("watts")
        tidy(ax, theme)

        axes = []
        for n, (key, colour) in enumerate(zip(order, theme["series"], strict=False)):
            sub = fig.add_subplot(grid[2 * n : 2 * n + 2, 1], sharex=axes[0] if axes else None)
            axes.append(sub)
            got = slice_at(key, start, 24.0)
            if got is not None:
                sub.plot(got[0], got[1], color=colour, linewidth=0.9)
            sub.set_title(key, loc="left", color=colour, fontsize=9)
            sub.set_ylabel("watts")
            tidy(sub, theme)
            if n < len(order) - 1:
                sub.tick_params(labelbottom=False)
            else:
                sub.set_xlabel("hours")
        axes[0].annotate(
            "the same day, submetered",
            xy=(0.0, 1.55),
            xycoords="axes fraction",
            color=theme["ink"],
            fontsize=9.5,
            fontweight="semibold",
        )

    render("smartnialm", build)


def do_plaid():
    from nilmframe.readers import PLAID

    root = CACHE / "plaid"
    recs = list(PLAID(root / "csv", [root / "meta_2017.json", root / "meta_2014.json"]))
    classes = by_class([r for r in recs if r.appliance])
    SUMMARY["plaid"] = {"rate_hz": recs[0].fs, "appliances": sorted(classes)}

    def build(fig, theme):
        grid = grid_for(fig)
        pick = [classes[k] for k in sorted(classes)][:3]
        detail_pair(fig, grid, theme, pick[2], f"{pick[2].appliance}, 30 kHz", 60.0)
        channel_stack(
            fig,
            grid,
            theme,
            [(r.appliance, r) for r in pick],
            "three appliance classes, 30 kHz",
            f0=60.0,
        )

    render("plaid", build)


def do_whited():
    from nilmframe.readers import WHITED

    recs = list(WHITED(CACHE / "whited/flac"))
    classes = by_class(recs)
    SUMMARY["whited"] = {"rate_hz": recs[0].fs, "appliances": sorted(classes)}

    def build(fig, theme):
        grid = grid_for(fig)
        pick = [classes[k] for k in sorted(classes)][:3]
        detail_pair(fig, grid, theme, pick[1], f"{pick[1].appliance}, 44.1 kHz", 50.0)
        channel_stack(
            fig,
            grid,
            theme,
            [(r.appliance, r) for r in pick],
            "three appliance classes, 44.1 kHz",
            f0=50.0,
        )

    render("whited", build)


def do_hifda():
    from nilmframe.readers import HIFDA

    recs = list(HIFDA(CACHE / "hifda/1310.72ms_window_dataset"))
    classes = by_class(recs)
    SUMMARY["hifda"] = {"rate_hz": recs[0].fs, "appliances": sorted(classes)}

    def build(fig, theme):
        grid = grid_for(fig)
        pick = [classes[k] for k in sorted(classes) if k != "empty grid"][:3]
        top, _ = detail_pair(fig, grid, theme, pick[1], f"{pick[1].appliance}, 100 kHz", 50.0)
        top.annotate(
            "50 Hz removed by the 300 Hz high-pass",
            xy=(0.02, 0.88),
            xycoords="axes fraction",
            color=theme["muted"],
            fontsize=8,
        )
        channel_stack(
            fig,
            grid,
            theme,
            [(r.appliance, r) for r in pick],
            "three appliance classes, 100 kHz",
            f0=50.0,
        )

    render("hifda", build)


# --------------------------------------------------------------------------- #


def do_vi_signatures():
    """The V-I trajectory figure -- a *feature*, so it lives with the features."""
    from nilmframe.readers import PLAID

    root = CACHE / "plaid"
    classes = by_class(
        [
            r
            for r in PLAID(root / "csv", [root / "meta_2017.json", root / "meta_2014.json"])
            if r.appliance
        ]
    )
    pick = [classes[k] for k in sorted(classes)][:3]

    def build(fig, theme):
        grid = GridSpec(1, 3, figure=fig, wspace=0.32)
        for n, (rec, colour) in enumerate(zip(pick, theme["series"], strict=False)):
            ax = fig.add_subplot(grid[0, n])
            _, a, b = cycle_window(rec, 60.0, cycles=1.0)
            v = rec.signals["v"][a:b].astype(float)
            i = rec.signals["i"][a:b].astype(float)
            peak = float(np.abs(i).max()) or 1.0
            ax.plot(np.r_[v, v[:1]], np.r_[i, i[:1]] / peak, color=colour, linewidth=1.5)
            ax.set_title(rec.appliance, loc="left", color=colour)
            ax.set_xlabel("volts")
            ax.set_ylim(-1.15, 1.15)
            if n == 0:
                ax.set_ylabel("current ÷ peak")
            tidy(ax, theme)
            ax.annotate(
                f"peak {peak:.2f} A",
                xy=(0.5, 0.03),
                xycoords="axes fraction",
                ha="center",
                color=theme["muted"],
                fontsize=8,
            )

    render("vi-signatures", build, size=(11.0, 3.2))


if __name__ == "__main__":
    names = ["blond", "fired", "ukdale", "smartnialm", "plaid", "whited", "hifda", "vi_signatures"]
    for name in sys.argv[1:] or names:
        fn = globals().get(f"do_{name}")
        if fn is None:
            print(f"  ?? no plotter for {name}")
            continue
        try:
            fn()
        except Exception as exc:
            print(f"  !! {name}: {type(exc).__name__}: {exc}")
    path = OUT / "summary.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(SUMMARY)
    path.write_text(json.dumps(existing, indent=1, default=str))
