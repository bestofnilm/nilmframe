"""iAWE reader.

One house in Delhi, logged for about ten weeks in 2013 at one sample a second.
Its value here is the mains: unlike UK-DALE, which publishes apparent power and
leaves reactive power to be recovered from the power triangle, iAWE's smart
meter *measures* reactive power directly, alongside apparent power, voltage,
current, frequency and power factor. That makes it the only public
low-frequency set besides AMPds2 where the (dP, dQ) plane is measured rather
than inferred.

The house is metered on two feeds. NILMTK calls both of them site meters, and
the whole-house load is their sum: neither alone is the building.

The sub-meters are logged sparsely -- the clothes iron has a few thousand
samples across two months -- because the plug loggers record while an appliance
draws and fall silent otherwise. A minute with no reading is therefore read as
off, which is the only interpretation the record supports, and the coverage of
each channel is reported so the assumption can be checked rather than assumed.

The file is the NILMTK HDF5 distribution (``iawe.h5``), whose appliance names
live in a pickled metadata blob on the building group.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from nilmframe.store.schema import ChannelKind, Recording

__all__ = ["IAWE_SITE_METERS", "IAWE"]

logger = logging.getLogger(__name__)

#: Both feeds are site meters; the building load is their sum.
IAWE_SITE_METERS = (1, 2)


class IAWE:
    """Read the iAWE HDF5 distribution.

    Args:
        path: the ``iawe.h5`` file.
        rate_hz: uniform output rate. The native rate is 1 Hz; one minute
            matches the low-frequency protocols this is usually compared under.
        dataset: name recorded on every channel.

    Example:
        >>> from nilmframe.readers.iawe import IAWE_SITE_METERS
        >>> IAWE_SITE_METERS
        (1, 2)
    """

    def __init__(self, path: str | Path, *, rate_hz: float = 1.0 / 60.0,
                 dataset: str = "iawe") -> None:
        self.path = Path(path)
        self.rate_hz = rate_hz
        self.dataset = dataset

    def appliances(self) -> dict[int, str]:
        """Meter number -> appliance name, from the building's metadata blob."""
        import h5py

        with h5py.File(self.path, "r") as fh:
            meta = pickle.loads(fh["building1"].attrs["metadata"])
        out: dict[int, str] = {}
        for entry in meta.get("appliances", []):
            name = (entry.get("original_name") or entry.get("type") or "unknown")
            for m in entry.get("meters", []):
                out[int(m)] = str(name).replace(" ", "_")
        return out

    def _grid(self, index: np.ndarray, values: np.ndarray,
              t0: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Average samples into bins, and report which bins had any sample."""
        step = 1.0 / self.rate_hz
        idx = np.floor((index - t0) / step).astype(np.int64)
        keep = (idx >= 0) & (idx < n)
        idx, values = idx[keep], values[keep]
        total = np.bincount(idx, weights=values, minlength=n)
        count = np.bincount(idx, minlength=n)
        return np.divide(total, count, out=np.zeros(n), where=count > 0), count > 0

    def __iter__(self) -> Iterator[Recording]:
        import pandas as pd

        step = 1.0 / self.rate_hz
        names = self.appliances()
        with pd.HDFStore(self.path, "r") as store:
            site = []
            for m in IAWE_SITE_METERS:
                frame = store.select(f"/building1/elec/meter{m}")
                site.append(frame)
            starts = [f.index[0].value / 1e9 for f in site]
            stops = [f.index[-1].value / 1e9 for f in site]
            t0 = float(np.floor(min(starts) / step) * step)
            n = int((max(stops) - t0) / step) + 1

            def col(frame, quantity):
                for c in frame.columns:
                    if str(c).find(quantity) >= 0:
                        return frame[c].to_numpy(np.float64)
                raise KeyError(quantity)

            p_total = np.zeros(n)
            q_total = np.zeros(n)
            seen = np.zeros(n, bool)
            for frame in site:
                stamps = frame.index.astype("int64").to_numpy() / 1e9
                p_grid, ok = self._grid(stamps, col(frame, "active"), t0, n)
                q_grid, _ = self._grid(stamps, col(frame, "reactive"), t0, n)
                p_total += p_grid
                q_total += q_grid
                seen |= ok
            yield Recording(
                dataset=self.dataset, house="house_1", session="run_0",
                kind=ChannelKind.MAINS, appliance=None, instance_id=None,
                signals={"p": p_total, "q": q_total},
                fs=self.rate_hz, t0=t0,
                meta={"meters": list(IAWE_SITE_METERS), "coverage": float(seen.mean()),
                      "reactive": "measured"},
            )

            for m, name in sorted(names.items()):
                try:
                    frame = store.select(f"/building1/elec/meter{m}")
                except KeyError:
                    logger.warning("iAWE: meter%d is not in the file", m)
                    continue
                stamps = frame.index.astype("int64").to_numpy() / 1e9
                try:
                    values = col(frame, "active")
                except KeyError:
                    values = frame.iloc[:, 0].to_numpy(np.float64)
                grid, ok = self._grid(stamps, values, t0, n)
                yield Recording(
                    dataset=self.dataset, house="house_1", session="run_0",
                    kind=ChannelKind.SUBMETER, appliance=name,
                    instance_id=f"house_1:{name}",
                    signals={"p": grid}, fs=self.rate_hz, t0=t0,
                    meta={"meter": m, "coverage": float(ok.mean())},
                )
