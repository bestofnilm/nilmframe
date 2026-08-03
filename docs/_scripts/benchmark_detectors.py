"""Benchmark the seven event detectors on real submetered data.

The ground truth is built the way the event-detection literature builds it, and
the construction is the part worth scrutinising. There is no annotator marking
switching instants in these corpora; what there is, is a submeter per appliance.
So a true event is a *submeter* crossing its on/off threshold, and the detectors
only ever see the *aggregate*. That is exactly the task -- find the transition in
the sum, having been told nothing about the parts -- and it is why the numbers
here are lower than a paper reporting on hand-marked BLUED transients.

Two consequences to keep in mind when reading the output:

* **Unmetered load counts against you.** Anything switching that has no submeter
  is a real transition in the aggregate with no ground-truth event behind it, so
  it is scored a false positive. In a house with nine submeters and forty
  appliances that is most of the false positives, and no detector can fix it.
* **Simultaneous switches merge.** Two appliances starting within the tolerance
  are one detectable edge but two true events, so recall is capped below one.

Run on a machine that has already fetched the subsets; writes figures next to the
dataset ones and a table to stdout.
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
import torch
from matplotlib.gridspec import GridSpec

from nilmframe.eval import EventCounts, EventF1, EventTimingError
from nilmframe.nn import (
    ActiveSectionDetector,
    AdaptiveThresholdDetector,
    CusumDetector,
    GLRDetector,
    GoodnessOfFitDetector,
    MultivariateDetector,
    ZScoreDetector,
)

logging.disable(logging.WARNING)

CACHE = Path.home() / "nf-work/cache"
OUT = Path.home() / "nf-work/plots"
OUT.mkdir(parents=True, exist_ok=True)

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


def detectors(rate_hz: float) -> dict[str, object]:
    """The seven, with windows scaled to the sampling rate.

    A window is a duration, not a sample count: 32 samples is four minutes at
    UK-DALE's 1/6 Hz and two milliseconds at BLOND's. Every detector here is
    configured from seconds so the comparison is between algorithms rather than
    between accidental window lengths.
    """
    window = max(4, round(60.0 * rate_hz))  # one minute of context
    gap = max(2, round(120.0 * rate_hz))  # events at least 2 min apart
    return {
        "ZScore": ZScoreDetector(window=window, threshold=4.0, min_gap=gap, min_delta=30.0),
        "CUSUM": CusumDetector(window=window, threshold=300.0, drift=20.0),
        "GLR": GLRDetector(pre=window, post=window, threshold=60.0, min_gap=gap, min_delta=30.0),
        "GoodnessOfFit": GoodnessOfFitDetector(
            pre=window, post=window, threshold=300.0, min_gap=gap, min_delta=30.0
        ),
        "Adaptive": AdaptiveThresholdDetector(window=window, base=30.0, scale=4.0, min_gap=gap),
        "Multivariate": MultivariateDetector(
            GLRDetector(pre=window, post=window, threshold=60.0, min_gap=gap, min_delta=30.0),
            rule="any",
            align=max(1, gap // 4),
        ),
        "ActiveSection": ActiveSectionDetector(
            window=window, threshold=40.0, min_steady=max(2, gap // 2)
        ),
    }


def truth_from_submeters(submeters: dict[str, np.ndarray], on_watts: float = 20.0) -> np.ndarray:
    """True event samples: any submeter crossing its on/off threshold.

    Args:
        submeters: appliance name to its power series, all on one clock.
        on_watts: the power above which an appliance counts as running.

    Returns:
        Sorted sample indices, one per transition, merged when two appliances
        switch on the same sample.
    """
    marks: set[int] = set()
    for series in submeters.values():
        on = series > on_watts
        changed = np.flatnonzero(on[1:] != on[:-1]) + 1
        marks.update(int(k) for k in changed)
    return np.array(sorted(marks), dtype=np.int64)


def as_instants(mask: torch.Tensor) -> torch.Tensor:
    """Collapse a dense mask to the leading edge of each run.

    :class:`ActiveSectionDetector` answers with intervals rather than instants,
    which is the point of it. Scoring every active sample as a separate detection
    would report one washing-machine ramp as two hundred false alarms and say
    nothing about whether the ramp was found.
    """
    previous = torch.nn.functional.pad(mask[:-1], (1, 0), value=False)
    return mask & ~previous


def score(mask: torch.Tensor, truth: np.ndarray, tolerance: int) -> dict[str, float]:
    """Event F1, timing error and raw counts for one detector on one channel."""
    actual = torch.from_numpy(truth)
    f1, timing, counts = (
        EventF1(tolerance=tolerance),
        EventTimingError(tolerance=tolerance),
        EventCounts(tolerance=tolerance),
    )
    for metric in (f1, timing, counts):
        metric.update(mask, actual)
    precision, recall = f1.precision_recall()
    raw = counts.compute()
    return {
        "f1": float(f1.compute()),
        "precision": float(precision),
        "recall": float(recall),
        "timing": float(timing.compute()),
        "tp": int(raw["tp"]),
        "fp": int(raw["fp"]),
        "fn": int(raw["fn"]),
    }


# --------------------------------------------------------------------------- #


def _runs(reader) -> tuple[list[tuple], dict[str, list[tuple]]]:
    """Every run of every channel as ``(t0, series, fs)``, split mains from submeters."""
    mains: list[tuple] = []
    subs: dict[str, list[tuple]] = {}
    for rec in reader:
        row = (rec.t0, np.asarray(rec.signals["p"], dtype=np.float64), rec.fs)
        if rec.kind.value == "mains":
            mains.append(row)
        else:
            subs.setdefault(rec.appliance, []).append(row)
    return mains, subs


def _align(mains: list[tuple], subs: dict[str, list[tuple]], hours: float) -> dict:
    """Put the aggregate and its submeters on one absolute-time window.

    Runs are cut per channel on that channel's own gaps, so run *k* of the mains
    and run *k* of a submeter are not the same stretch of time -- picking the
    longest of each independently lands them in different months. The window is
    therefore chosen as the mains run that the most submeters actually overlap.
    """

    def span(row):
        t0, series, fs = row
        return t0, t0 + series.size / fs

    def overlap_score(row):
        a, b = span(row)
        return sum(
            1
            for runs in subs.values()
            for other in runs
            if min(b, span(other)[1]) - max(a, span(other)[0]) > 3600
        )

    best = max(mains, key=lambda row: (overlap_score(row), row[1].size))
    t0, series, rate = best
    count = min(series.size, int(hours * 3600 * rate))
    window = series[:count]

    aligned: dict[str, np.ndarray] = {}
    for name, runs in subs.items():
        padded = np.zeros(count)
        covered = 0
        for other_t0, other, _ in runs:
            shift = round((other_t0 - t0) * rate)
            source_from = max(0, -shift)
            target_from = max(0, shift)
            length = min(other.size - source_from, count - target_from)
            if length > 0:
                padded[target_from : target_from + length] = other[
                    source_from : source_from + length
                ]
                covered += length
        # A channel with no overlap contributes only zeros, which would look like
        # an appliance that never ran rather than one that was not measured.
        if covered > count // 100:
            aligned[name] = padded
    return {"rate": rate, "mains": window, "submeters": aligned}


def load_ukdale(hours: float = 24.0) -> dict:
    from nilmframe.readers import UKDALE

    mains, subs = _runs(UKDALE(CACHE / "ukdale/low_freq", high_freq=False, houses=[1]))
    if not mains:
        raise RuntimeError("no UK-DALE aggregate in the cache")
    return {"name": "UK-DALE house 1", **_align(mains, subs, hours)}


def load_refit(hours: float = 24.0) -> dict:
    from nilmframe.readers import REFIT

    mains, subs = _runs(REFIT(CACHE / "refit", houses=[1]))
    if not mains:
        raise RuntimeError("no REFIT aggregate in the cache")
    return {"name": "REFIT house 1", **_align(mains, subs, hours)}


#: The knob each detector is tuned on, and the grid searched over it.
KNOBS = {
    "ZScore": ("threshold", [1.5, 2.0, 3.0, 4.0, 6.0]),
    "CUSUM": ("threshold", [200.0, 500.0, 1000.0, 2000.0, 4000.0]),
    "GLR": ("threshold", [20.0, 60.0, 150.0, 400.0, 1000.0]),
    "GoodnessOfFit": ("threshold", [100.0, 300.0, 800.0, 2000.0, 5000.0]),
    "Adaptive": ("scale", [2.0, 3.0, 4.0, 6.0, 9.0]),
    "Multivariate": (None, [None]),
    "ActiveSection": ("threshold", [20.0, 40.0, 80.0, 150.0, 300.0]),
}


def detect(name: str, detector, mains: torch.Tensor) -> torch.Tensor:
    """One detector's event instants for a power series."""
    with torch.no_grad():
        if name == "Multivariate":
            # Nothing else is measured here, so the second channel is the first
            # difference: a crude stand-in for reactive power that at least gives
            # the rule something independent to agree with.
            gradient = torch.diff(mains, prepend=mains[:1]).abs()
            mask = detector(torch.stack([mains, gradient]))
        else:
            mask = detector(mains)
    return as_instants(mask) if name == "ActiveSection" else mask


def run(dataset: dict, tolerance_s: float = 60.0) -> list[dict]:
    """Tune on the first half of the window, report on the second.

    Every detector here has a threshold, and a comparison at thresholds somebody
    guessed is a comparison of the guesses. De Baets et al. (2017) make the case
    for tuning them and for asking whether the tuning transfers -- so each is
    swept on the first half and scored on the second, which it has not seen.
    """
    rate = dataset["rate"]
    tolerance = max(1, round(tolerance_s * rate))
    mains = torch.from_numpy(dataset["mains"]).float()
    middle = mains.numel() // 2

    truth = truth_from_submeters(dataset["submeters"])
    tune_truth = truth[truth < middle]
    report_truth = truth[truth >= middle] - middle
    tune_mains, report_mains = mains[:middle], mains[middle:]

    rows = []
    for name in detectors(rate):
        knob, grid = KNOBS[name]
        best_value, best_f1 = None, -1.0
        for value in grid:
            candidate = detectors(rate)[name]
            if knob is not None:
                setattr(candidate, knob, value)
            found = detect(name, candidate, tune_mains)
            f1 = score(found, tune_truth, tolerance)["f1"]
            if f1 > best_f1:
                best_value, best_f1 = value, f1

        final = detectors(rate)[name]
        if knob is not None:
            setattr(final, knob, best_value)
        result = score(detect(name, final, report_mains), report_truth, tolerance)
        result["detector"] = name
        result["knob"] = None if knob is None else f"{knob}={best_value:g}"
        result["tuned_f1"] = best_f1
        rows.append(result)
    return rows


def figure(dataset: dict, rows: list[dict], name: str) -> None:
    rate = dataset["rate"]
    truth = truth_from_submeters(dataset["submeters"])
    mains = torch.from_numpy(dataset["mains"]).float()
    best = max(rows, key=lambda r: r["f1"])
    detector = detectors(rate)[best["detector"]]
    if best.get("knob"):
        knob, value = best["knob"].split("=")
        setattr(detector, knob, float(value))
    found = np.flatnonzero(detect(best["detector"], detector, mains).numpy())

    for mode, theme in THEMES.items():
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
            }
        )
        fig = plt.figure(figsize=(11.0, 5.6), dpi=155)
        grid = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.28, width_ratios=[1.35, 1.0])

        # Left: the aggregate with true events and one detector's calls. Zoomed to
        # the busiest few hours -- at a whole day the marks are a solid bar and
        # say nothing about whether any individual event was found.
        ax = fig.add_subplot(grid[:, 0])
        span = int(3 * 3600 * rate)
        if truth.size and mains.numel() > span:
            counts = [
                ((truth >= s) & (truth < s + span)).sum()
                for s in range(0, mains.numel() - span, max(1, span // 4))
            ]
            begin = int(np.argmax(counts)) * max(1, span // 4)
        else:
            begin = 0
        stop = min(mains.numel(), begin + span)
        minutes = (np.arange(begin, stop) - begin) / rate / 60
        ax.plot(minutes, mains.numpy()[begin:stop], color=theme["series"][0], linewidth=1.0)
        for k in truth[(truth >= begin) & (truth < stop)]:
            ax.axvline((k - begin) / rate / 60, color=theme["series"][2], alpha=0.55, linewidth=1.0)
        for k in found[(found >= begin) & (found < stop)]:
            ax.axvline(
                (k - begin) / rate / 60,
                color=theme["series"][1],
                alpha=0.85,
                linewidth=1.0,
                linestyle="--",
            )
        ax.set_title(
            "{} — {}, F1 {:.2f}".format(dataset["name"], best["detector"], best["f1"]),
            loc="left",
            color=theme["ink"],
        )
        ax.set_xlabel("minutes (busiest 3 h of the day)")
        ax.set_ylabel("watts")
        ax.grid(True, alpha=0.45, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.annotate(
            "solid: submeter transitions   dashed: detected",
            xy=(0.01, 0.97),
            xycoords="axes fraction",
            fontsize=8,
            color=theme["muted"],
            va="top",
        )

        # Right top: F1 per detector.
        order = sorted(rows, key=lambda r: r["f1"])
        bar = fig.add_subplot(grid[0, 1])
        bar.barh(
            [r["detector"] for r in order],
            [r["f1"] for r in order],
            color=theme["series"][0],
            height=0.6,
        )
        bar.set_xlim(0, 1)
        bar.set_title("event F1", loc="left", color=theme["ink"])
        bar.grid(True, axis="x", alpha=0.45, linewidth=0.6)
        bar.set_axisbelow(True)

        # Right bottom: precision against recall, which F1 hides.
        pr = fig.add_subplot(grid[1, 1])
        pr.scatter(
            [r["recall"] for r in rows],
            [r["precision"] for r in rows],
            color=theme["series"][1],
            s=42,
            zorder=3,
        )
        # Detectors that land on the same point would print their names on top of
        # each other; stagger the labels rather than lose one.
        placed: dict[tuple[int, int], int] = {}
        for row in sorted(rows, key=lambda r: -r["precision"]):
            cell = (round(row["recall"], 2), round(row["precision"], 2))
            nth = placed.get(cell, 0)
            placed[cell] = nth + 1
            pr.annotate(
                row["detector"],
                (row["recall"], row["precision"]),
                textcoords="offset points",
                xytext=(6, 4 - 11 * nth),
                fontsize=7.5,
                color=theme["muted"],
            )
        pr.set_xlim(-0.02, 1.02)
        pr.set_ylim(-0.02, 1.02)
        pr.set_xlabel("recall")
        pr.set_ylabel("precision")
        pr.set_title("the trade-off F1 averages away", loc="left", color=theme["ink"])
        pr.grid(True, alpha=0.45, linewidth=0.6)
        pr.set_axisbelow(True)

        fig.savefig(OUT / f"detectors-{name}-{mode}.png", bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
    print(f"  figure: detectors-{name}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or ["ukdale", "refit"]
    summary = {}
    for key in wanted:
        loader = {"ukdale": load_ukdale, "refit": load_refit}[key]
        try:
            dataset = loader()
        except Exception as exc:
            print(f"  !! {key}: {type(exc).__name__}: {exc}")
            continue
        truth = truth_from_submeters(dataset["submeters"])
        hours = dataset["mains"].size / dataset["rate"] / 3600
        print(
            f"\n=== {dataset['name']} | {len(dataset['submeters'])} submeters "
            f"| {truth.size} true events | {hours:.0f} h at {dataset['rate']:.4f} Hz"
        )
        rows = run(dataset)
        head = ("detector", "F1", "prec", "rec", "timing", "tp", "fp", "fn", "tuned")
        print(
            f"  {head[0]:<14}{head[1]:>7}{head[2]:>7}{head[3]:>7}{head[4]:>8}"
            f"{head[5]:>6}{head[6]:>6}{head[7]:>6}  {head[8]}"
        )
        for row in sorted(rows, key=lambda r: -r["f1"]):
            seconds = row["timing"] / dataset["rate"]
            print(
                f"  {row['detector']:<14}{row['f1']:>7.3f}{row['precision']:>7.3f}"
                f"{row['recall']:>7.3f}{seconds:>7.1f}s{row['tp']:>6}{row['fp']:>6}"
                f"{row['fn']:>6}  {row.get('knob') or '-'}"
            )
        figure(dataset, rows, key)
        summary[key] = {"dataset": dataset["name"], "n_true": int(truth.size), "rows": rows}
    (OUT / "detector_benchmark.json").write_text(json.dumps(summary, indent=1))
