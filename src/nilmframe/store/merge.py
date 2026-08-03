"""Merging stores under explicit rules.

Combining corpora is easy to do and easy to do wrongly, so the rules are
arguments rather than assumptions:

``require``
    Axes that must already agree. Merging two corpora whose supply voltages
    differ, without saying so, produces a store where the same appliance has two
    current scales and no model can tell why.
``taxonomy``
    Appliance labels, reconciled. The merged label space is the union of the
    inputs, so two names for one appliance become two classes unless something
    maps them together. A :class:`~nilmframe.taxonomy.Taxonomy` does that knowing
    which corpus each label came from, and can be inspected before the merge runs.
``rename``
    A flat ``{from: to}`` override applied after the taxonomy, for the one-off
    case that does not deserve a table.
``normalize_voltage``
    Rescale every channel to one supply level. Voltage is scaled up and current
    down by the same factor, so **active power is unchanged** -- this harmonises
    the signal shape without relabelling the load.

What is deliberately *not* harmonised is the sampling rate. Resampling a waveform
to a common rate would throw away the higher-rate corpus's detail for no gain,
because the thing that actually makes rates comparable is cycle alignment, which
happens later in the view and costs nothing here. See :mod:`nilmframe.compat`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nilmframe.store.reader import Store
from nilmframe.store.writer import StoreWriter

if TYPE_CHECKING:  # pragma: no cover - typing only, and taxonomy imports Store
    from nilmframe.taxonomy import Taxonomy

__all__ = ["merge_stores"]

_AXIS_COLUMN = {"fs": "fs", "f0": "f0", "quantities": "quantities", "dataset": "dataset"}


def merge_stores(
    sources: Sequence[str | Path | Store],
    dst: str | Path,
    *,
    require: Sequence[str] = (),
    taxonomy: Taxonomy | Mapping[str, Sequence[str]] | None = None,
    rename: Mapping[str, str] | None = None,
    normalize_voltage: float | None = None,
    prefix_with_dataset: bool = True,
    overwrite: bool = False,
) -> Store:
    """Combine several stores into one.

    Args:
        sources: stores, or paths to them.
        dst: destination directory.
        require: axes that must agree across the inputs before merging is allowed.
            Any of ``fs``, ``f0``, ``quantities``, ``dataset``, ``voltage``. The
            useful ones are ``voltage`` and ``f0``; ``quantities`` almost always
            varies in a real store, because a house's waveform mains and its
            meter channels sit side by side, and a view selects between them
            rather than being broken by them.
        taxonomy: a :class:`~nilmframe.taxonomy.Taxonomy`, or an alias table to
            build one from. Resolves each label knowing which corpus it came
            from, and fills the ``category`` column. Labels it does not recognise
            are left alone — check :meth:`~nilmframe.taxonomy.Taxonomy.unmapped`
            first if that matters.
        rename: flat alias map applied to every channel and activation, after the
            taxonomy. Use it for one-offs; use ``taxonomy`` for a vocabulary.
        normalize_voltage: rescale to this RMS supply level in volts. Current is
            scaled inversely, so active power is preserved.
        prefix_with_dataset: prefix channel ids with their dataset, so two corpora
            that both call a channel ``house_1-mains`` do not collide.
        overwrite: replace an existing store at ``dst``.

    Returns:
        The merged :class:`~nilmframe.store.Store`.

    Raises:
        ValueError: when an axis listed in ``require`` disagrees, naming what differs.

    Example:
        >>> import tempfile, pathlib
        >>> dst = pathlib.Path(tempfile.mkdtemp()) / 'merged'
        >>> merged = nf.merge_stores([store], dst,
        ...                           rename={'microwave': 'oven'},
        ...                           normalize_voltage=230.0)
        >>> merged.appliances
        ['fridge', 'kettle', 'laptop', 'oven']
        >>> merged.manifest['merge_rules']['normalize_voltage']
        230.0
    """
    from nilmframe.taxonomy import Taxonomy as _Taxonomy

    stores = [s if isinstance(s, Store) else Store(s) for s in sources]
    if not stores:
        raise ValueError("merge_stores() needs at least one source")

    _check_required(stores, require, normalize_voltage)

    if taxonomy is not None and not isinstance(taxonomy, _Taxonomy):
        taxonomy = _Taxonomy(taxonomy)
    rename = dict(rename or {})
    dst = Path(dst).expanduser()

    # The appliance table is keyed by label, not by (dataset, label), so a
    # taxonomy override that depends on the corpus cannot be honoured here.
    # Channels can be and are -- this only sets thresholds and categories.
    thresholds: dict[str, float] = {}
    categories: dict[str, str] = {}
    for store in stores:
        for _, row in store.appliance_table.iterrows():
            name = _relabel(row["appliance"], None, taxonomy, rename)
            thresholds[name] = float(row["on_threshold_w"])
            category = str(row["category"])
            if category in ("", "unknown") and taxonomy is not None:
                category = taxonomy.category(name)
            categories[name] = category

    writer = StoreWriter(
        dst,
        source="merge of " + ", ".join(str(s.path) for s in stores),
        appliance_thresholds=thresholds,
        appliance_categories=categories,
        overwrite=overwrite,
    )

    for store in stores:
        for _, row in store.channels.iterrows():
            channel_id = row["channel_id"]
            new_id = f"{row['dataset']}-{channel_id}" if prefix_with_dataset else channel_id
            quantities = str(row["quantities"]).split(",")
            scale = _voltage_scale(store, channel_id, quantities, normalize_voltage)

            _emit(writer, store, row, new_id, quantities, taxonomy, rename, scale)

    writer.close()
    merged = Store(dst)
    merged.manifest["merged_from"] = [str(s.path) for s in stores]
    merged.manifest["merge_rules"] = {
        "require": list(require),
        # The resolved map, not the Taxonomy object: a manifest has to be JSON and
        # has to stay readable years later, when the table has moved on.
        "taxonomy": taxonomy.as_dict(*stores) if taxonomy is not None else {},
        "rename": rename,
        "normalize_voltage": normalize_voltage,
    }
    _rewrite_manifest(dst, merged.manifest)
    return Store(dst)


# --------------------------------------------------------------------------- #


def _check_required(stores, require, normalize_voltage) -> None:
    from nilmframe.compat import compatibility

    if not require:
        return
    report = compatibility(*stores, deep="voltage" in require)
    problems = []
    for axis_name in require:
        key = "supply voltage" if axis_name == "voltage" else _AXIS_COLUMN.get(axis_name, axis_name)
        try:
            axis = report.axis(key)
        except KeyError as exc:
            raise ValueError(
                f"unknown axis {axis_name!r}; expected any of {sorted({*_AXIS_COLUMN, 'voltage'})}"
            ) from exc
        if axis.varies:
            if axis_name == "voltage" and normalize_voltage:
                continue  # about to be harmonised anyway
            problems.append(f"{axis_name} differs across the inputs: {axis.values}")
    if problems:
        raise ValueError("cannot merge under the requested rules:\n  - " + "\n  - ".join(problems))


def _voltage_scale(store, channel_id, quantities, target) -> float:
    """Factor to bring a channel to the target supply level. 1.0 when not applicable.

    The head it measures is trimmed to a whole number of mains cycles. RMS over a
    partial cycle is biased -- the sine does not average out -- and the bias lands
    directly in the scale, so every sample of the merged channel inherits it.
    """
    if not target or "v" not in quantities:
        return 1.0

    row = store.channel(channel_id)
    fs, f0, available = float(row["fs"]), float(row["f0"]), int(row["n_samples"])
    wanted = min(available, int(fs))  # about a second
    if np.isfinite(f0) and f0 > 0:
        period = fs / f0
        cycles = max(1, int(wanted // period))
        wanted = min(available, round(cycles * period))

    head = store.read_window(channel_id, "v", 0, wanted).astype(np.float64)
    vrms = float(np.sqrt(np.mean(np.square(head))))
    return target / vrms if vrms > 1.0 else 1.0


def _relabel(appliance, dataset, taxonomy, rename):
    """One label through the taxonomy, then through the explicit overrides.

    In that order, so ``rename`` is the last word: it is what you reach for when
    the table is wrong about your data, and an override that the table could
    overrule would not be one.
    """
    if appliance is None or _is_missing(appliance):
        return None
    if taxonomy is not None:
        resolved = taxonomy.resolve(appliance, dataset)
        if resolved is not None:
            appliance = resolved
    return rename.get(appliance, appliance)


def _emit(writer, store, row, new_id, quantities, taxonomy, rename, scale) -> None:
    """Write one channel into the merged store, copying signals when untouched."""
    from nilmframe.store.schema import Activation, ChannelKind, Recording

    dataset = row["dataset"]
    appliance = _relabel(row["appliance"], dataset, taxonomy, rename)

    signals = {}
    for quantity in quantities:
        array = np.array(store.signal(row["channel_id"], quantity))
        if scale != 1.0:
            # Voltage up, current down by the same factor: the load's power is a
            # property of the load, not of the supply it was measured on.
            array = array * (scale if quantity == "v" else 1.0 / scale)
        signals[quantity] = array.astype(np.float32)

    activations = [
        Activation(_relabel(a.appliance, dataset, taxonomy, rename), int(a.on), int(a.off))
        for a in store.activations_for(row["channel_id"]).itertuples()
    ]

    writer.add(
        Recording(
            dataset=dataset,
            house=row["house"],
            session=row["session"],
            kind=ChannelKind(row["kind"]),
            appliance=appliance,
            brand=None if _is_missing(row["brand"]) else row["brand"],
            instance_id=row["instance_id"],
            signals=signals,
            fs=float(row["fs"]),
            f0=None if not np.isfinite(row["f0"]) else float(row["f0"]),
            t0=float(row["t0"]),
            activations=activations,
            meta={"merged_from": row["channel_id"], "voltage_scale": scale},
        ),
        channel_id=new_id,
    )


def _is_missing(value) -> bool:
    """pandas spells "absent" as NaN, which is truthy."""
    return value is None or (isinstance(value, float) and not np.isfinite(value))


def _rewrite_manifest(path: Path, manifest: dict) -> None:
    import json

    (Path(path) / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
