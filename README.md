# nilmframe

PyTorch-native end-to-end non-intrusive load monitoring (NILM), built around three ideas:

1. **One canonical store.** Tabular metadata plus memory-mapped signal arrays. Nothing is
   materialised until a window is requested, so datasets larger than RAM are ordinary.
2. **Many rate views over the same store.** The low-frequency series a 1 Hz model sees is
   *derived from* the same waveform a 16 kHz model sees, by the same code, over the same window
   boundaries. Comparing low- against high-frequency modelling is a config flag, not a second
   ingestion pipeline — and there is no preprocessing confound left to argue about.
3. **Honest outputs.** Separate presence logits and non-negative power, the *measured* aggregate
   as an input rather than the sum of labels, and a conservation term tying both heads to a
   physical quantity.

```python
import torch
import nilmframe as nf

store = nf.Store("~/.nilm/store")                       # built once by `nilmframe convert`
view  = nf.HighFreqView(n_cycles=20, cycle_size=128, align="fitps")
split = nf.LeaveBrandOut(test_size=0.2, seed=0).apply(store)

train = nf.WindowDataset(store, split.train, view=view, targets=("presence", "power"))
dl    = torch.utils.data.DataLoader(train, batch_size=64, num_workers=8, pin_memory=True)
```

For interactive work there is `Measurement` — one measurement as an object rather than a dict
of tensors. It is a lens over the same lazy store, and it is immutable, so chaining off it
cannot disturb anything:

```python
m = store.measurement(channel_id, seconds=2)
m.aligned(cycle_size=128).lowpass(8).harmonics(6)     # chain
(fridge + kettle).active_power(per_component=True)    # superpose, keep parts separable
m.plot(); m.vi_image(); m.batch()                     # look at it, or hand it to a model
```

## Install

```bash
pip install -e ".[all]"      # or: pip install -e .   for the light core
```

The core depends only on `torch`, `numpy`, `pandas` and `pyarrow`. There is **no compiled
extension**: cycle alignment is pure PyTorch, so it is batched, runs on the GPU as part of the
model, and installs everywhere. Optional extras: `readers` (FLAC for WHITED), `eval`
(torchmetrics), `train` (Lightning), `dev` (pytest, ruff).

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Package skeleton, CI config, licence | done |
| 1 | `cycle_align` in torch + golden tests | done |
| 2 | Canonical store, PLAID/WHITED converters, `nilmframe convert` | done |
| 3 | `WindowDataset`, views, transforms, split protocols | done |
| 4 | Heads, `DisaggLoss`, metrics, honest `predict()` | done |
| 5 | `MixAggregate` augmentation, open-set channel and metrics | done |
| 6 | UK-DALE converter, `LowFreqView`, LF/HF sweep | done |

All six phases are implemented and green, plus an audit pass that closed five gaps between the plan
and the build (CI, event detectors, `train`/`evaluate`, the TorchScript contract test, and the delete
list). `plan.html` has a **What Was Actually Built** section recording the seven points where the
implementation departs from it and why — including the one specified item deliberately left undone.

Run `pytest` for the full suite (351 tests, ~20 s); `pytest -m "not slow"` skips the
sweep. The suite builds miniature datasets in the real on-disk formats -- PLAID
CSVs, WHITED stereo FLAC, UK-DALE `.dat` plus `mains.flac` -- so parsing,
calibration and window arithmetic are exercised rather than mocked.

## Getting the data

The public corpora are large and you rarely want all of one. `nilmframe fetch`
works out which remote files a subset needs and pulls only those -- five minutes
of BLOND's 8.9 TB, one hour of UK-DALE's 16 kHz mains, one appliance out of
WHITED's 2.1 GB archive, four HIFDA recordings out of an archive holding 770,612
files. Planning is separate from fetching, so `--dry-run` prints the bill first.

```bash
nilmframe fetch whited --cache ~/data/WHITED --appliances Kettle --dry-run
```

Four of the five are HTTP hosts that honour range requests, and a zip keeps its
directory at the end -- so a single member is reachable without the archive.
BLOND comes over its publisher's FTP delivery server, where `MLSD` costs a plan
and `REST` resumes a transfer. UK-DALE's meter channels also stop decompressing
once the readings pass the end of a `--to` bound: a day in early 2013 costs 13 MB
against 3.6 GB for the archive.

## Combining datasets

Merging corpora is the normal case. `compatibility()` reports what varies — sampling rate,
mains frequency, supply voltage, quantities, vocabulary — and which of those actually break
which view. Most do not: cycle alignment resamples every mains cycle onto a fixed grid, so a
6 kHz 60 Hz recording and a 44.1 kHz 50 Hz one come out the same shape. What alignment does
*not* fix is supply voltage and naming, so those are explicit rules:

```bash
nilmframe compat store/plaid store/whited
nilmframe merge store/plaid store/whited --dst store/combined \
    --require voltage --rename refrigerator=fridge --normalize-voltage 230
```

`--normalize-voltage` scales voltage up and current down by the same factor, so every load's
active power is unchanged. A `--require`d axis that disagrees refuses the merge rather than
quietly producing a corpus nobody can reason about.

## The low- versus high-frequency experiment

```bash
nilmframe fetch   ukdale --cache ~/data/UK-DALE --channels 1 5 \
                        --from 2015-01-20 --to 2015-01-21 --max-hf-files 2
nilmframe convert ukdale --src ~/data/UK-DALE/low_freq --dst ~/.nilm/ukdale \
                        --high-freq-root ~/data/UK-DALE/high_freq --rate-hz 0.1667
nilmframe sweep configs/lf_vs_hf.yaml --store ~/.nilm/ukdale --out runs/lf_vs_hf
```

Every arm shares one store, one split, one loss and one metric set; they differ
only in the `view`. The `highfreq-unaligned` arm is the control that separates
*having the waveform* from *aligning it*, and both high-frequency arms consume
identical windows so the comparison is not confounded by how much signal each saw.
The run writes `results.csv` and a `manifest.json` recording the store's content
hash, the split, the specs and the environment.

## Layout

```
src/nilmframe/
  store/     schema.py  writer.py  reader.py     canonical store: parquet metadata + memmapped signals
  readers/   plaid.py  whited.py  ukdale.py      source datasets -> Recording records
             blond.py  hifda.py  smartnialm.py
  sources/   _zip.py  _ftp.py  fetch.py         fetch a subset of a corpus, not the corpus
             ukdale.py  blond.py  hifda.py  smartnialm.py
  measurement.py                                 one measurement as an object (interactive)
  compat.py                                      what varies across corpora, and what it breaks
  data/      dataset.py  views.py  windows.py    torch Dataset, rate views, window index
             splits.py  mixing.py                leakage-safe protocols, on-the-fly aggregation
  nn/        align.py  repr.py  backbones.py     cycle alignment, representations, encoders
             heads.py  losses.py  task.py        presence/power heads, conservation loss
  eval/      metrics.py  protocols.py  report.py detection / power / joint / open-set metrics
  cli.py                                         fetch, convert, train, evaluate, sweep
tests/reference/fitps.h                          the original C++, kept as a test oracle
```

The pre-refactor code has been deleted per the delete list in `plan.html`, now that everything on
its keep-list is ported and tested. Git history retains it at the initial-import commit
(`git show 3c5d359`). The one survivor is `tests/reference/fitps.h`: it is no longer a dependency,
only the oracle the torch cycle alignment is checked against.

## Why not the previous design

The predecessor stored a Python list of sample objects and manipulated it eagerly. An audit
(reproduced in `plan.html`) found that no module imported, `n_components` returned the number of
time samples so `submetered()` always returned an empty dataset, per-component power silently
returned aggregate power, the FITPS interpolation used a constant weight instead of the fractional
part (31–259× error, varying per cycle with zero-crossing phase), and models were scored after being
rescaled by `Y_true.sum(1)` — a ground-truth total. Those are design consequences, not typos, and
this package is the redesign.

## Licence

Apache-2.0. See `LICENSE`.
