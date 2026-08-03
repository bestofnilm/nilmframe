"""Builders for miniature datasets in the *real on-disk formats*.

The converters are tested against actual PLAID CSV files and actual WHITED stereo
FLAC files rather than against mocks, so the parsing paths -- column order,
calibration scaling, filename fields, activation annotations -- are genuinely
exercised. Everything here is a few hundred kilobytes and builds in well under a
second, which is what makes it usable as a CI fixture.

Signals are physically plausible: a mains sine at a configurable off-nominal
frequency, and appliance currents with characteristic harmonic content, so that
alignment, power computation and target construction all see something with the
right shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = [
    "APPLIANCE_PROFILES",
    "appliance_current",
    "mains_voltage",
    "make_plaid",
    "make_whited",
]

#: (rms amperes, harmonic amplitudes relative to the fundamental, phase lag)
APPLIANCE_PROFILES: dict[str, tuple[float, tuple[float, ...], float]] = {
    "kettle": (9.0, (1.0, 0.02, 0.01), 0.02),  # near-resistive
    "fridge": (0.9, (1.0, 0.12, 0.30), 0.55),  # inductive motor
    "laptop": (0.4, (1.0, 0.55, 0.40), 0.35),  # switched-mode supply
    "fan": (0.6, (1.0, 0.10, 0.06), 0.45),
    "microwave": (6.0, (1.0, 0.30, 0.18), 0.30),
    "vacuum": (4.5, (1.0, 0.25, 0.15), 0.40),
}


def mains_voltage(n: int, fs: float, f0: float = 50.0, vrms: float = 230.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return (vrms * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def appliance_current(
    appliance: str,
    n: int,
    fs: float,
    f0: float = 50.0,
    *,
    scale: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """A periodic current with the harmonic signature of ``appliance``."""
    rms, harmonics, lag = APPLIANCE_PROFILES[appliance]
    t = np.arange(n, dtype=np.float64) / fs
    wave = np.zeros(n, dtype=np.float64)
    for order, amplitude in enumerate(harmonics, start=1):
        wave += amplitude * np.sin(2 * np.pi * f0 * order * t - lag * order)
    wave /= np.sqrt(np.mean(wave**2)) or 1.0
    out = rms * np.sqrt(2) * scale * wave
    if seed is not None:
        out = out + np.random.default_rng(seed).normal(0, 0.002 * rms, n)
    return out.astype(np.float32)


def make_plaid(
    root: str | Path,
    *,
    fs: float = 6000.0,
    f0: float = 50.1,
    seconds: float = 1.0,
    submetered: tuple[tuple[str, str, str], ...] = (
        # (appliance, brand, model) -- two instances of some classes so that
        # brand-disjoint and instance-disjoint splits have something to separate.
        ("kettle", "acme", "k100"),
        ("kettle", "acme", "k100"),
        ("kettle", "globex", "g7"),
        ("fridge", "acme", "f1"),
        ("fridge", "initech", "cool9"),
        ("laptop", "globex", "l3"),
        ("laptop", "globex", "l3"),
        ("fan", "initech", "breeze"),
    ),
    n_aggregate: int = 2,
) -> tuple[Path, Path]:
    """Write a miniature PLAID-format dataset.

    Returns:
        ``(csv_dir, metadata_path)``, ready to pass to
        :class:`~nilmframe.readers.PLAID`.
    """
    root = Path(root)
    csv_dir = root / "CSV"
    csv_dir.mkdir(parents=True, exist_ok=True)
    n = int(fs * seconds)
    voltage = mains_voltage(n, fs, f0)
    metadata: dict[str, dict] = {}
    rid = 0

    for appliance, brand, model in submetered:
        rid += 1
        current = appliance_current(appliance, n, fs, f0, seed=rid)
        # PLAID's column order is current first, voltage second -- getting this
        # backwards silently swaps the signals, so the fixture encodes it.
        np.savetxt(csv_dir / f"{rid}.csv", np.c_[current, voltage], delimiter=",", fmt="%.6f")
        metadata[str(rid)] = {
            "header": {"sampling_frequency": f"{int(fs)}Hz", "collection_time": "2013-06-01"},
            "appliance": {
                "type": appliance.replace("_", " ").title(),
                "brand": brand,
                "model": model,
            },
        }

    aggregate_sets = [("kettle", "fridge"), ("laptop", "fan", "fridge")]
    for k in range(n_aggregate):
        rid += 1
        members = aggregate_sets[k % len(aggregate_sets)]
        total = np.zeros(n, dtype=np.float32)
        entries = []
        for j, appliance in enumerate(members):
            on = int(n * 0.1 * (j + 1))
            off = int(n * (0.6 + 0.15 * j))
            current = appliance_current(appliance, n, fs, f0, seed=100 + rid * 10 + j)
            gate = np.zeros(n, dtype=np.float32)
            gate[on:off] = 1.0
            total += current * gate
            entries.append(
                {"type": appliance.replace("_", " ").title(), "on": f"[{on}]", "off": f"[{off}]"}
            )
        np.savetxt(csv_dir / f"{rid}.csv", np.c_[total, voltage], delimiter=",", fmt="%.6f")
        metadata[str(rid)] = {
            "header": {"sampling_frequency": f"{int(fs)}Hz", "collection_time": "2013-06-02"},
            "appliances": entries,
        }

    metadata_path = root / "meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=1))
    return csv_dir, metadata_path


def make_whited(
    root: str | Path,
    *,
    fs: int = 44100,
    f0: float = 50.0,
    seconds: float = 0.2,
    entries: tuple[tuple[str, str, str, str], ...] = (
        # (appliance, model, region, kit)
        ("Microwave", "mwA", "DE", "MK1"),
        ("Microwave", "mwB", "DE", "MK2"),
        ("Vacuum", "vacA", "DE", "MK1"),
        ("Fan", "fanA", "AT", "MK3"),
    ),
) -> Path:
    """Write a miniature WHITED-format dataset (stereo FLAC). Returns the directory."""
    import soundfile as sf

    from nilmframe.readers.whited import CALIBRATION

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n = int(fs * seconds)

    for k, (appliance, model, region, kit) in enumerate(entries):
        factors = CALIBRATION[kit]
        voltage = mains_voltage(n, fs, f0)
        current = appliance_current(appliance.lower(), n, fs, f0, seed=k)
        # WHITED stores normalised PCM; the reader multiplies the kit factors back
        # in, so the fixture must divide them out to round-trip.
        stereo = np.c_[voltage / factors["volt"], current / factors["amp"]]
        stereo = np.clip(stereo, -1.0, 1.0).astype(np.float32)
        sf.write(
            root / f"{appliance}_{model}_{region}_{kit}_20150101.flac",
            stereo,
            fs,
            subtype="PCM_24",
            format="FLAC",
        )
    return root


def make_ukdale(
    root: str | Path,
    *,
    houses: tuple[int, ...] = (1, 2),
    period_s: float = 6.0,
    hours: float = 2.0,
    gap_at: float | None = 0.5,
    gap_s: float = 3600.0,
    t0: float = 1_352_164_036.0,
    high_freq: bool = True,
    hf_fs: int = 16000,
    hf_seconds: float = 1.0,
    seed: int = 0,
) -> Path:
    """Write a miniature dataset in UK-DALE's on-disk format.

    Per house: ``labels.dat``, a mains ``channel_1.dat`` that is the exact sum of
    its submeters, and one ``channel_N.dat`` per appliance. The mains being an
    exact sum is what lets a test assert that sibling-derived targets reconstruct
    it, which is the property the LF/HF comparison rests on.

    ``gap_at`` inserts a dropout part-way through, so gap-splitting is exercised.
    """
    from nilmframe.readers.whited import CALIBRATION  # noqa: F401  (kept for symmetry)

    root = Path(root)
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / period_s)

    for house in houses:
        house_dir = root / f"house_{house}"
        house_dir.mkdir(parents=True, exist_ok=True)

        appliances = ["kettle", "fridge", "laptop"][: 2 + (house % 2)]
        timestamps = t0 + np.arange(n) * period_s
        if gap_at is not None:
            cut = int(n * gap_at)
            timestamps[cut:] += gap_s

        submeters = {}
        for k, appliance in enumerate(appliances):
            rms, _, _ = APPLIANCE_PROFILES[appliance]
            watts = np.zeros(n)
            # A few duty cycles per appliance, offset so they overlap partially.
            duty = rng.uniform(0.15, 0.4)
            phase = (k + 1) / (len(appliances) + 1)
            cycle = np.linspace(0, 1, n, endpoint=False)
            on = ((cycle * (3 + k) + phase) % 1.0) < duty
            watts[on] = rms * 230.0 * rng.uniform(0.9, 1.1, on.sum())
            submeters[appliance] = watts

        mains = sum(submeters.values()) + rng.uniform(20, 40, n)  # standing load

        labels = ["1 aggregate"]
        np.savetxt(house_dir / "channel_1.dat", np.c_[timestamps, mains], fmt="%.6f")
        for k, (appliance, watts) in enumerate(submeters.items(), start=2):
            labels.append(f"{k} {appliance}")
            np.savetxt(house_dir / f"channel_{k}.dat", np.c_[timestamps, watts], fmt="%.6f")
        (house_dir / "labels.dat").write_text("\n".join(labels) + "\n")

        if high_freq:
            import soundfile as sf

            m = int(hf_fs * hf_seconds)
            voltage = mains_voltage(m, hf_fs, 50.0, vrms=230.0)
            current = appliance_current("kettle", m, hf_fs, 50.0, seed=seed + house)
            # Undo the reader's calibration so the round trip lands in volts/amps.
            stereo = np.c_[voltage / 415.0, current / 30.0]
            sf.write(
                house_dir / "mains.flac",
                np.clip(stereo, -1, 1).astype(np.float32),
                hf_fs,
                subtype="PCM_24",
                format="FLAC",
            )
            (house_dir / "mains.dat").write_text(f"{timestamps[0]:.6f} 0.0\n")

    return root
