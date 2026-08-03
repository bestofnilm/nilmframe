---
myst:
  html_meta:
    description: What each public NILM corpus contains, at what resolution, and what it can and cannot be used for.
---

# A tour of the datasets

Nine corpora ship with readers. They are not interchangeable, and the differences
matter more than the file formats do: some give you an aggregate to disaggregate
and some only give you isolated appliances, some are waveforms and some are power
readings, and two of them have a property that quietly invalidates a whole class
of analysis if you do not know about it.

This chapter is what you would otherwise learn by downloading all nine. Every
figure below was produced by the readers in this package, from a small fetched
subset — the code is in `docs/_scripts/make_dataset_plots.py`.

The figures all have the same shape and all show **raw signals**: on the left one
channel in detail, on the right three more channels of the same corpus. Waveform
panels show three whole mains cycles at that dataset's own line frequency, so the
horizontal axis means the same thing everywhere. Nothing here is a derived
representation — for V–I trajectories and the rest, see {doc}`representations`.

## At a glance

```{list-table}
:header-rows: 1
:widths: 14 12 12 20 22 20

* - Corpus
  - Rate
  - Signals
  - Channels
  - Scale
  - Good for
* - {ref}`BLOND <ds-blond>`
  - 50 kHz / 6.4 kHz
  - v, i
  - 3-phase aggregate **and** 15×6 submetered sockets
  - 8.9 TB, 213 days, 1 site
  - waveform disaggregation with real ground truth
* - {ref}`FIRED <ds-fired>`
  - 8 kHz / 2 kHz / 1 Hz
  - v, i **and** p
  - 3-phase aggregate **and** 21 plug meters
  - 3.2 TB, 52 days, 1 apartment
  - waveform disaggregation in a home
* - {ref}`UK-DALE <ds-ukdale>`
  - 16 kHz / 1/6 Hz
  - v, i **and** p
  - aggregate **and** submeters, at two rates
  - ~3.6 TB, 4 years, 5 houses
  - comparing what sampling rate is worth
* - {ref}`SmartNIALMeter <ds-snm>`
  - 0.2 Hz
  - p
  - aggregate **and** submeters
  - 45 GB, 2 years, 20 buildings
  - low-rate disaggregation across many buildings
* - {ref}`REFIT <ds-refit>`
  - 1/8 Hz
  - p
  - aggregate **and** submeters
  - 2.2 GB, 2 years, 6 of 20 houses
  - transfer across many homes
* - {ref}`UCI household <ds-uci>`
  - 1/60 Hz
  - p
  - aggregate **and** 3 circuits
  - 20 MB, 4 years, 1 house
  - the standard low-frequency baseline
* - {ref}`PLAID <ds-plaid>`
  - 30 kHz
  - v, i
  - submeters only (plus a few aggregates)
  - 695 MB, 1793 recordings
  - appliance classification from signatures
* - {ref}`WHITED <ds-whited>`
  - 44.1 kHz
  - v, i
  - submeters only
  - 2.1 GB, 1339 activations
  - start-up transients, region/kit transfer
* - {ref}`HIFDA <ds-hifda>`
  - 100 kHz
  - v, i
  - submeters only (plus an empty-grid class)
  - 4.4 GB, 750 recordings
  - high-frequency signatures **only** — see the warning
```

"Submeters only" is the distinction to read first. A corpus without an aggregate
cannot pose the disaggregation problem at all; what it supports is
*classification* — given one appliance's waveform, name it — or, with
{class}`~nilmframe.data.MixAggregate`, a synthetic aggregate you built yourself.
Three of the nine are like this. BLOND, FIRED, UK-DALE, SmartNIALMeter, REFIT and
the UCI household set all come with a measured aggregate; PLAID, WHITED and HIFDA
do not.

(ds-blond)=
## BLOND

```{image} ../_static/datasets/blond-light.png
:class: only-light
:alt: BLOND: 50 kHz mains voltage and current, and the currents of three metered sockets at 6.4 kHz.
```

```{image} ../_static/datasets/blond-dark.png
:class: only-dark
:alt: BLOND: 50 kHz mains voltage and current, and the currents of three metered sockets at 6.4 kHz.
```

An office building measured for 213 days: three-phase mains at 50 kHz from one
"CLEAR" unit, and fifteen "MEDAL" units of six metered sockets each at 6.4 kHz.
The left panels are one phase of the mains: the current is visibly nothing like a
sine, because it is the sum of dozens of switch-mode supplies. The right panels
are the current of three sockets behind it, at the MEDAL units' lower rate.

Those socket traces look like noise because they nearly are: a dev board, a
monitor and a battery charger draw tens of milliamps, measured on a range sized
for a whole circuit. That is worth seeing before you plan an experiment on
individual sockets — an office is mostly small loads, and the per-socket
signal-to-noise at 6.4 kHz is the constraint you will actually hit.

```{code-block} python
from nilmframe.readers import BLOND

reader = BLOND.download("~/.cache/nilmframe/blond",
                        units=["clear", "medal-1"], days=["2016-09-30"],
                        max_files=1, max_seconds=10)
```

Nuances worth knowing before you use it:

- **The labels are a history.** `appliance_log.json` records each MEDAL's socket
  configuration stamped with the moment it took effect; sockets were re-used over
  seven months. The reader takes the newest entry at or before each recording's
  clock. Reading the first entry mislabels whole weeks, and every label still
  looks like a real appliance.
- **Sockets carry a DC offset that must be removed.** The offset subtracted during
  recording was the converter's nominal midpoint, and the true zero drifts. On
  `medal-1` socket 1 the residual is −0.21 A against an RMS of 0.21 A: read it
  as-is and four-fifths of the "current" is offset. The reader removes the mean
  per mains cycle, as the dataset's authors prescribe. `remove_offset=False`
  gives the stored samples if you want them.
- **A MEDAL belongs to a phase.** Each unit sits on `L1`, `L2` or `L3`, recorded
  on the channel as `meta["phase"]`. Only that phase's mains current contains it,
  so matching a mains window to all fifteen units is wrong.
- One CLEAR file is 5 minutes and 118 MB; a day of the whole rig is 42 GB.

(ds-fired)=
## FIRED

```{image} ../_static/datasets/fired-light.png
:class: only-light
:alt: FIRED: a plug meter's voltage and current at 2 kHz, and the apartment's aggregate and two appliances at 1 Hz.
```

```{image} ../_static/datasets/fired-dark.png
:class: only-dark
:alt: FIRED: a plug meter's voltage and current at 2 kHz, and the apartment's aggregate and two appliances at 1 Hz.
```

Fifty-two days of a three-room German apartment, fully labelled: a three-phase
smart meter at 8 kHz, twenty-one plug-level meters at 2 kHz, and 50 Hz and 1 Hz
power summaries of the same measurements. BLOND's shape in a home rather than an
office, with the waveform on both sides of the problem. The right panels show a
day of it — the fridge's compressor duty cycle, and the two times the heat lamp
ran.

```{code-block} python
from nilmframe.readers import FIRED

reader = FIRED.download("~/.cache/nilmframe/fired",
                        resolution="1Hz", meters=["smartmeter001", "powermeter08"])
```

- **It is stored as audio.** Every file is WavPack inside Matroska, which is a
  good choice — lossless, and it compresses these signals well — but decoding
  needs `ffmpeg` on `PATH`. The containers carry `TIMESTAMP` and `CHANNEL_TAGS`,
  so a file needs no index to be placed on the clock and its quantities name
  themselves.
- **The waveform current is in milliamperes.** Read it as amperes and the
  apartment draws a hundred kilowatts. The dataset checks itself here: scaling by
  a thousand makes the waveform's mean `v * i` agree with the 1 Hz summary's own
  active power to under a percent — 58.1 W against 57.8 W on the first ten
  minutes of phase L1. The reader applies it.
- **Two plug meters were installed backwards.** `deviceMapping.json` flags them
  with `flip`; their current is negated, or the appliance appears to generate
  power.
- **Fetching is over rsync**, which is what its authors publish. That needs an
  `rsync` binary — the only dataset here with a non-Python requirement.
- Three tiers: 1.7 GB for all the 1 Hz summaries, 80 GB adding 50 Hz, 3.2 TB
  adding the waveforms.

(ds-ukdale)=
## UK-DALE

```{image} ../_static/datasets/ukdale-light.png
:class: only-light
:alt: UK-DALE: 16 kHz mains voltage and current, and the same house's aggregate and submeters at 1/6 Hz.
```

```{image} ../_static/datasets/ukdale-dark.png
:class: only-dark
:alt: UK-DALE: 16 kHz mains voltage and current, and the same house's aggregate and submeters at 1/6 Hz.
```

The dataset this package's low-versus-high-frequency experiment is built on,
because it is the one that carries both views of the *same house on the same
clock*: 16 kHz voltage and current for house 1's mains, and per-appliance meters
at roughly one reading every six seconds. The left panels are three cycles of the
mains; the right panels are six hours of the same house at 1/6 Hz, with the
aggregate above two submeters.

```{code-block} python
from nilmframe.readers import UKDALE

reader = UKDALE.download("~/.cache/nilmframe/ukdale",
                         houses=[1], channels=[1, 5],
                         time_range=(1421784000, 1421870400), max_hf_files=2)
```

- **Only house 1 has waveforms.** Houses 2–5 are meter readings only.
- **A waveform takes its session from the meter run containing it**, which is how
  a mains window finds the submeters that label it. Fetch waveforms without the
  meter channels covering the same hours and every window lands in `hf_only` with
  no targets — see {doc}`downloading`.
- **The meters drift and drop out.** The reader forward-fills onto a uniform grid
  because a power meter reports a step function, and splits on gaps longer than
  `max_gap_s` rather than filling them, which would manufacture hours of an
  appliance being on.

(ds-snm)=
## SmartNIALMeter

```{image} ../_static/datasets/smartnialm-light.png
:class: only-light
:alt: SmartNIALMeter: a building's aggregate over a day, and the submeters behind it.
```

```{image} ../_static/datasets/smartnialm-dark.png
:class: only-dark
:alt: SmartNIALMeter: a building's aggregate over a day, and the submeters behind it.
```

Twenty buildings, up to two years each, one reading every five seconds: a smart
meter plus a dedicated sensor on every appliance behind it. This is the picture of
the disaggregation problem as most papers pose it — the 5 kW block in the
aggregate between hours 12 and 14 is the boiler, and the submeter panel says so.

At 0.2 Hz there is no waveform. Cycle alignment, V–I trajectories and harmonics do
not apply. What you get instead is *many buildings*, which is what you need to
hold some out.

```{code-block} python
from nilmframe.readers import SmartNIALM

reader = SmartNIALM.download("~/.cache/nilmframe/snm",
                             buildings=[1], appliances=["boiler", "freezer"])
```

- **The aggregate is the file called `cii-adapter`.** It is named for the
  interface it was read through — the smart meter's Consumer Information
  Interface — not for what it measures. It is the one file in all twenty
  buildings. Treat it as an appliance and the building's whole consumption enters
  your label space as a device.
- **The columns differ from file to file, and this is where the corpus is easiest
  to lose.** A single-phase appliance publishes `Active Power`; a three-phase one
  publishes `Active Power L1..L3`, which are halves of one machine and must be
  summed; and the smart meter publishes *no power column at all* — only `Voltage`,
  `Current` and `Power Factor` per phase, whose product summed over phases is the
  aggregate. Reading the literal `Active Power` column gets you the single-phase
  appliances and neither the aggregate nor the largest loads.
- **Real installations have outages.** Building 1's channels split into 37–53 runs
  each over the record; the longest unbroken run is 221 days. Runs are cut per
  channel on that channel's own gaps, so the aggregate and its submeters do *not*
  share run boundaries — align on `t0`, not on session.
- Fetching is coarse: the release is a `.7z` in twelve solid blocks, so one file
  costs its whole block. Ask for a building at once, not a file at a time.

(ds-refit)=
## REFIT

Twenty UK households at eight-second resolution for about two years each: mains
plus up to nine appliances per home. Its value is the *number of houses* — most
fully submetered corpora are one building, and "transfers across twenty homes" is
a different claim from "fits one".

```{code-block} python
from nilmframe.readers import REFIT

reader = REFIT.download("~/.cache/nilmframe/refit", houses=[1, 2])
```

- **The columns do not say what they measure.** The CSV header is
  `Aggregate,Appliance1..Appliance9`, and `Appliance4` is a washer dryer in house
  1 and something else in house 2. The reader carries the published per-house
  mapping, so the store gets `washer_dryer` rather than `appliance4`.
- **The published names disagree with themselves.** `Dishwaser` appears
  alongside `Dishwasher`, `Firdge` alongside `Fridge`, `Fridge-Freezer` alongside
  `Fridge_Freezer`, and qualifiers like `Freezer(garage)` sit beside `Freezer(1)`.
  Left alone that is 40 appliance classes where there are 34, and the duplicates
  are exactly the common appliances. The reader reconciles them and keeps the
  original in `meta["original_name"]`.
- **The unit of fetching is a house**, because a house is a file — about 400 MB
  holding its mains and every appliance. The Zenodo record carries 6 of the 20;
  the full release sits in a repository that blocks some hosts.
- `Issues` flags rows the cleaning could not reconcile. They are kept by default;
  `drop_issues=True` removes them.

(ds-uci)=
## UCI household

One French house, four years, one reading a minute, 20 MB. The smallest thing
here and the most benchmarked — a great deal of published forecasting and
disaggregation work uses it, so it is worth having despite being a single home.

```{code-block} python
from nilmframe.readers import UCIHousehold

reader = UCIHousehold.download("~/.cache/nilmframe/uci")
```

- **The units are not watts.** `Global_active_power` is in *kilowatts* and the
  three sub-meterings in *watt-hours per minute*. At face value the aggregate and
  its submeters sit three orders of magnitude apart. The reader converts both.
- **The submeters do not sum to the aggregate, by design.** They cover three
  circuits — kitchen, laundry, water heater plus air conditioning — and the rest
  of the house is unmetered, usually the larger part. This is a *partially*
  submetered aggregate, which is a different problem from a fully submetered one.
- **Missing values are `?`**, about 1.25% of rows. They become gaps rather than
  zeros, because a zero here reads as an appliance that was off.

(ds-plaid)=
## PLAID

```{image} ../_static/datasets/plaid-light.png
:class: only-light
:alt: PLAID: a hairdryer's voltage and current at 30 kHz, and the currents of three appliance classes.
```

```{image} ../_static/datasets/plaid-dark.png
:class: only-dark
:alt: PLAID: a hairdryer's voltage and current at 30 kHz, and the currents of three appliance classes.
```

1793 short recordings of individual appliances at 30 kHz, collected in US homes —
note the 120 V, 60 Hz mains in the left panels, where every other corpus here is
230 V at 50 Hz. The right panels are three classes' current at the same scale of
time: a compact fluorescent lamp draws a narrow spike twice a cycle, a fridge's
motor a rounded and phase-shifted sine, a hairdryer very nearly the voltage's own
shape. Those differences are what {doc}`representations` turns into features.

```{code-block} python
from nilmframe.readers import PLAID

reader = PLAID.download("~/.cache/nilmframe/plaid", limit=50)
```

- **The annotations come in two files that partition the corpus.** `meta_2017.json`
  describes 719 recordings and `meta_2014.json` the other 1074, with no overlap.
  Passing one reads 40% of the release. The reader takes a list.
- **The house is `location`, not the collection date.** There are 55 sites; the
  collection campaign has two values, so deriving a house from the date leaves
  {class}`~nilmframe.data.LeaveHouseOut` nothing to partition.
- Mostly submetered, single-appliance recordings; a minority are aggregates
  annotated with per-appliance on/off sample indices, which the reader turns into
  {class}`~nilmframe.store.Activation` intervals.

(ds-whited)=
## WHITED

```{image} ../_static/datasets/whited-light.png
:class: only-light
:alt: WHITED: a kettle's voltage and current at 44.1 kHz, and the currents of three appliance classes.
```

```{image} ../_static/datasets/whited-dark.png
:class: only-dark
:alt: WHITED: a kettle's voltage and current at 44.1 kHz, and the currents of three appliance classes.
```

1339 activations at 44.1 kHz, each capturing an appliance being *switched on* —
which is what distinguishes it from PLAID's steady-state recordings, and what
makes it the corpus for start-up transients. The panels above show the steady part
of three of them. Recorded in several regions with three measurement kits, so it
also supports asking whether a model transfers across meters.

```{code-block} python
from nilmframe.readers import WHITED

reader = WHITED.download("~/.cache/nilmframe/whited", appliances=["Kettle"])
```

- **Submeters only.** No aggregate anywhere in the corpus.
- **The measurement kit sets the calibration.** `MK1`, `MK2` and `MK3` have
  different scaling factors and the kit is in the filename; a recording whose kit
  has no factor is skipped rather than silently mis-scaled.
- **Only the flat files are the corpus.** The archive also holds `Experiments/`
  (including runs with two appliances on at once), `notUsed/` and `MIXED/`, whose
  names look like corpus recordings but are not. The fetcher takes the 1339 flat
  files and says how many it left behind.

(ds-hifda)=
## HIFDA

```{image} ../_static/datasets/hifda-light.png
:class: only-light
:alt: HIFDA: a microwave at 100 kHz with a flat voltage channel, and the currents of three appliance classes.
```

```{image} ../_static/datasets/hifda-dark.png
:class: only-dark
:alt: HIFDA: a microwave at 100 kHz with a flat voltage channel, and the currents of three appliance classes.
```

The highest rate here: 100 kSPS for fourteen appliances, plus the empty grid
measured under the same conditions — a background class most submetered corpora
lack. The right panels show how different three classes look at that rate. The
same measurements are published windowed four ways, from 10.24 ms slices to whole
5.4-second activations; `--window` chooses.

:::{admonition} The grid fundamental is not in this data
:class: warning

Look at the voltage panel above: it is flat noise around zero, and that is not a
bug. HIFDA's voltage is band-limited to 300 Hz – 50 kHz and its current to roughly
30 Hz – 50 kHz, so the 50 Hz component is filtered out of **both**. The authors
were after the high-frequency signature.

The consequence is that this corpus does not support anything that needs the
fundamental. There are no zero crossings, so {class}`~nilmframe.data.HighFreqView`
with `align="fitps"` has nothing to align to. `v * i` is not the appliance's
power. Use it for high-frequency representations and classification, not for
disaggregation against an aggregate.
:::

```{code-block} python
from nilmframe.readers import HIFDA

reader = HIFDA.download("~/.cache/nilmframe/hifda",
                        appliances=["Microwave"], limit=10)
```

- **The files hold ADC volts, not amperes.** Samples span the converter's 0–3.3 V
  range; the reader applies the release's documented affine conversion, so the
  store gets physical units like every other corpus. `calibrate=False` gives the
  published numbers.
- **Submeters only**, plus the empty grid, which the reader makes a mains channel
  rather than an appliance called "empty grid".

## Which one to use

**You want to disaggregate a measured aggregate.** With waveforms: BLOND (an
office) or FIRED (an apartment). At meter rates: UK-DALE (five houses),
SmartNIALMeter (twenty buildings), REFIT (six of twenty houses) or the UCI
household set (one house, and only three circuits submetered). PLAID, WHITED and
HIFDA cannot pose the problem at all.

**You want a model that transfers between homes.** REFIT and SmartNIALMeter are
the two with enough separate buildings to hold some out.

**You want to classify appliances from their signatures.** PLAID and WHITED are
the standard pair — steady state and switch-on transient — and HIFDA adds a much
higher rate and a background class. {doc}`representations` covers turning those
waveforms into something a model can use.

**You want to know whether the sampling rate is worth it.** UK-DALE, because it is
the only one carrying both views of one house on one clock. That comparison is
{doc}`its own chapter <views>`.

**You want to combine several.** Read {doc}`combining` first: the corpora disagree
on mains frequency, supply voltage, sampling rate and vocabulary, and
{func}`nilmframe.compatibility` enumerates the disagreements before you merge.
