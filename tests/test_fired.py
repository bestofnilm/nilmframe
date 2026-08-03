"""FIRED: the parts of the format that are easy to read wrongly.

FIRED ships its waveforms as WavPack inside Matroska, so a test that means
anything has to go through a real container. These build one with ``ffmpeg`` and
read it back through the reader; without ``ffmpeg`` on PATH they skip rather than
assert something weaker.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nilmframe.readers import FIRED

FS = 2000
F0 = 50.0
MILLIAMPS = 1000.0

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FIRED is WavPack in Matroska; needs ffmpeg"
)


def _write_mkv(path: Path, volts: np.ndarray, milliamps: np.ndarray) -> None:
    """One two-channel WavPack stream in Matroska, tagged the way FIRED tags them.

    The quantity names live in ``CHANNEL_TAGS`` on the stream, which is what the
    reader looks for -- a file without it is skipped with a warning.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.empty(volts.size * 2, dtype=np.float32)
    interleaved[0::2] = volts.astype(np.float32)
    interleaved[1::2] = milliamps.astype(np.float32)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(FS),
            "-ac",
            "2",
            "-i",
            "pipe:0",
            "-c:a",
            "wavpack",
            "-metadata:s:a:0",
            "CHANNEL_TAGS=v,i",
            str(path),
        ],
        input=interleaved.tobytes(),
        check=True,
    )


def _tree(root: Path, meter: str, flip: bool, watts: float) -> None:
    """A FIRED root holding one plug meter drawing ``watts`` at unity power factor."""
    t = np.arange(FS, dtype=np.float64) / FS
    volts = 325.0 * np.sin(2 * np.pi * F0 * t)
    # A resistive load: current in phase with voltage, so mean(v * i) is the power.
    amps = (watts * 2 / 325.0) * np.sin(2 * np.pi * F0 * t)
    _write_mkv(
        root / "highFreq" / meter / f"{meter}_2020_06_14__00_00_00.mkv",
        volts,
        amps * MILLIAMPS,
    )
    info = root / "info"
    info.mkdir(parents=True, exist_ok=True)
    entry = {"appliances": ["baby heat lamp"], "phase": 2}
    if flip:
        entry["flip"] = True
    (info / "deviceMapping.json").write_text(json.dumps({meter: entry}))
    (info / "deviceInfo.json").write_text("{}")


def _active_power(root: Path, meter: str) -> float:
    records = list(FIRED(root, resolution="highFreq", meters=[meter]))
    assert records, "the reader found nothing"
    record = records[0]
    v = np.asarray(record.signals["v"], np.float64)
    i = np.asarray(record.signals["i"], np.float64)
    return float((v * i).mean())


@pytest.mark.parametrize("flip", [False, True])
def test_a_plug_meter_consumes_power_whatever_the_flip_flag_says(tmp_path, flip):
    """`flip` marks the installation, not the file, and must not negate the current.

    The published waveforms already have the correction applied. Negating them a
    second time turns the appliance into a generator, which is the bug this pins:
    measured against FIRED's own 1 Hz summary, powermeter08 reads +598 W as
    published and -598 W if the flag is obeyed, and the summary says +599 W.
    """
    _tree(tmp_path, "powermeter08", flip=flip, watts=600.0)
    assert _active_power(tmp_path, "powermeter08") == pytest.approx(600.0, rel=0.02)


def test_current_is_amperes_not_milliamperes(tmp_path):
    """The container carries milliamps; the store's schema is amperes."""
    _tree(tmp_path, "powermeter08", flip=False, watts=600.0)
    record = next(iter(FIRED(tmp_path, resolution="highFreq", meters=["powermeter08"])))
    irms = float(np.sqrt((np.asarray(record.signals["i"], np.float64) ** 2).mean()))
    # 600 W at 230 V RMS is about 2.6 A. A missing conversion would report 2600.
    assert 2.0 < irms < 4.0
