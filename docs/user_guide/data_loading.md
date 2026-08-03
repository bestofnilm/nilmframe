# Loading data

Public NILM datasets agree on almost nothing. PLAID ships CSV files with a JSON
metadata sidecar. WHITED ships stereo FLAC, one file per activation, with the
measurement kit encoded in the filename and a different calibration constant per
kit. UK-DALE ships tab-separated meter readings at roughly 1/6 Hz alongside a
separate tree of hour-long 16 kHz FLAC waveforms, and a per-house label file
mapping meter numbers to appliance names.

Writing a model against three of those directly means writing three loaders, three
sets of unit conventions, and three quiet opportunities to get a scale factor
wrong. The first thing nilmframe does is remove that: every dataset is read once
into one canonical store, and everything downstream reads stores.

## The canonical store

A store is a directory:

```text
mystore/
  manifest.json          format version, source, appliance thresholds
  channels.parquet       one row per channel: ids, rate, f0, length, checksum
  activations.parquet    on/off intervals, where the dataset provides them
  signals/
    house_1-kettle-v.npy
    house_1-kettle-i.npy
    ...
```

Metadata is tabular. Signals are plain `.npy`, opened with `mmap_mode="r"`, so
reading a window touches only the pages that window covers. The cost of a window
into a house-year is the cost of the window, not the cost of the year. Nothing is
materialised until asked for.

That is the whole trick, and it is why there is no `Dataset` subclass per corpus:
a corpus larger than memory is ordinary rather than special.

## Getting a public dataset

The three corpora below are large — UK-DALE alone is terabytes — and you rarely want
all of one. {doc}`downloading` covers fetching a slice: a day of meter readings, an
hour of waveform, one appliance. Each reader can do it and hand back a reader over
what it fetched:

```{code-block} python
from nilmframe.readers import UKDALE

reader = UKDALE.download("~/.cache/nilmframe/ukdale", houses=[1],
                         channels=[1, 5], max_hf_files=1)
```

The rest of this chapter assumes the files are already on disk, however they got
there.

## Reading a public dataset

Five readers ship. Each is an iterable of {class}`~nilmframe.store.Recording`
objects, and each pairs with {class}`~nilmframe.store.StoreWriter`:

```{code-block} python
from nilmframe.readers import PLAID, WHITED, UKDALE
from nilmframe.store import StoreWriter

with StoreWriter("stores/plaid", source="PLAID v1") as w:
    for rec in PLAID("PLAID/CSV", "PLAID/meta.json"):
        w.add(rec)
```

WHITED is the same shape. Pass `strict=True` the first time you read a download so
an unrecognised measurement kit is reported rather than silently skipped:

```{code-block} python
with StoreWriter("stores/whited") as w:
    for rec in WHITED("WHITED", strict=True):
        w.add(rec)
```

UK-DALE needs more care, because its two halves live in separate trees and because
reading it naively costs hours:

```{code-block} python
reader = UKDALE(
    "ukdale/low_freq",                   # meter readings
    high_freq_root="ukdale/high_freq",   # vi-*.flac waveforms
    houses=[1],
    max_hf_files=1,                      # one file is an hour at 16 kHz, ~200 MB
    max_seconds=60,                      # ...and you rarely need the whole hour
)
with StoreWriter("stores/ukdale") as w:
    for rec in reader:
        w.add(rec)
```

`max_seconds` bounds the *waveform* reads only. It deliberately does not touch the
meter channels: a second of a 1/6 Hz series is not a sample, and truncating one to
satisfy a waveform budget would collapse the run span the waveform files are matched
against. To bound the meter extent, use `time_range` with unix seconds — the files
are sorted, so it stops early rather than reading years and discarding them.

:::{admonition} UK-DALE calibration
:class: note

The stored PCM is normalised, so it must be scaled to volts and amperes. UK-DALE's
own `calibration_house_N.cfg` gives `volts_per_adc_step` for the ADC input, which is
upstream of the voltage divider — it does not convert the stored samples directly.
The shipped default `(388.45, 251.51)` was derived from house 1 by scaling channel 0
to 230 V RMS, then choosing the current scale so the waveform's active power matches
what the low-frequency meter reported over the same hour. Override it with
`hf_calibration` if you derive better constants for your houses.
:::

## Writing your own reader

There is no base class to subclass. A reader is anything that yields
{class}`~nilmframe.store.Recording` objects, so the smallest one is a generator
function:

```{doctest}
>>> import numpy as np
>>> from nilmframe.store import Recording, ChannelKind, StoreWriter, Store
>>> import tempfile, pathlib
>>>
>>> def my_reader(n=2000, fs=6000.0):
...     t = np.arange(n) / fs
...     v = 325.0 * np.sin(2 * np.pi * 50.0 * t)
...     i = 4.0 * np.sin(2 * np.pi * 50.0 * t)
...     yield Recording(
...         dataset="mine",
...         house="lab",
...         session="run1",
...         kind=ChannelKind.SUBMETER,
...         signals={"v": v.astype("float32"), "i": i.astype("float32")},
...         fs=fs,
...         f0=50.0,
...         appliance="heater",
...     )
>>>
>>> path = pathlib.Path(tempfile.mkdtemp()) / "mine"
>>> with StoreWriter(path, source="hand-written") as w:
...     for rec in my_reader():
...         _ = w.add(rec)      # returns the channel id it assigned
>>> Store(path).channels[["channel_id", "appliance", "fs", "f0"]]
             channel_id appliance      fs    f0
0  mine-lab-run1-heater    heater  6000.0  50.0
```

The channel id was built for you, from dataset, house, session and appliance. Ids are
derived rather than supplied so that two corpora can be merged without a collision —
see {doc}`combining`.

The writer is what enforces the invariants: it checks that every signal in a
recording has the same length, that `fs` and `f0` are positive, that a submeter
names an appliance and a mains channel does not, and it records a SHA-256 of every
signal file it writes.

Two fields carry more weight than they look like they do.

`session` groups channels that were recorded *at the same time in the same place*.
It is what lets the store answer "what else was running while this kettle was on",
which is what {meth}`~nilmframe.store.Store.siblings` uses to build per-appliance
truth for a mains window. Get it wrong and aggregate targets are silently wrong.

`instance_id`, if you set it, identifies a *physical unit* — this specific kettle,
not kettles in general. Splitting is done on instances rather than on classes, so
that a model is never tested on the same appliance it was trained on. If you leave
it unset the writer derives one from house and appliance, which is right for most
corpora and wrong for any dataset that moved one device between houses.

## Checking a store

{meth}`~nilmframe.store.Store.verify` re-hashes every signal file against the
checksum recorded when it was written:

```{doctest}
>>> import nilmframe as nf
>>> store = nf.example_store()
>>> store.verify()
[]
```

An empty list means every file matches. Anything else is a list of channel ids whose
signals have changed on disk since they were written — a truncated download, an
interrupted copy, or an edit nobody meant to make.

{meth}`~nilmframe.store.Store.describe` is the other thing to run on a fresh store:

```{doctest}
>>> store.describe()
   appliance  channels  instances  brands     hours
0     fridge         2          2       2  0.001111
1     kettle         2          2       2  0.001111
2     laptop         1          1       1  0.000556
3  microwave         1          1       1  0.000556
```

The columns to read carefully are `instances` and `brands`. Two channels of one
physical fridge is not two fridges, and a class present in only one brand cannot
support a claim about generalising across brands — see {doc}`splitting`. From the
command line the same thing is `nilmframe describe <store> --verify`.

## The example store

Everything in this guide runs against a generated corpus:

```{doctest}
>>> store = nf.example_store()
>>> store
Store('nilmframe-example-store', channels=8, appliances=4, datasets=['example'])
>>> cols = ["channel_id", "house", "kind", "appliance", "brand"]
>>> print(store.channels[cols].head(4).fillna("—").to_string())
       channel_id    house      kind appliance brand
0  house_1-kettle  house_1  submeter    kettle  acme
1  house_1-fridge  house_1  submeter    fridge  acme
2  house_1-laptop  house_1  submeter    laptop  acme
3   house_1-mains  house_1     mains         —     —
```

The dashes are this example filling in the blanks, not a value in the store: a
mains channel has no appliance and no brand, so those cells are missing. They are
filled here because pandas renders a missing value differently across its own
major versions — `None` on 2.x, `NaN` on 3.x — and an example that pins one of
them is an example that fails on the other.

Two houses, two brands, four appliance classes, and a real mains channel per house
whose waveform is the sum of its submeters plus noise. It is small — half a second
per channel at 6 kHz — but it has the structure that matters: mains and submeters,
more than one house, more than one brand, and a class that appears in only one house.
That is enough to demonstrate every protocol in {doc}`splitting` and to make every
example on these pages executable without a download.

```{doctest}
>>> len(store), store.appliances
(8, ['fridge', 'kettle', 'laptop', 'microwave'])
```
