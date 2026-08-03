# Event detection

Before a classifier ever runs, something has to decide *when* to run it. A house's
power series is mostly nothing happening, punctuated by switching events, and a model
applied uniformly across it spends almost all of its budget on windows where nothing
changed.

Event detection is also useful on its own — for cutting a long recording into
activations, for building a training set from an unlabelled corpus, and for the
event-based metrics in {doc}`evaluation`.

Every detector in {mod}`nilmframe.nn` answers the same question — *did something change
here* — and answers it in the same shape, so they are interchangeable and the code
around them never has to know which one it is holding.

## The interface

A detector is an `nn.Module`. Call it with a power envelope and it returns a boolean
mask of the same shape, true where an event was detected. It works on `(T,)` and on
`(B, T)` alike.

```{doctest}
>>> import torch, nilmframe as nf
>>> import nilmframe.nn as nn_
>>> _ = torch.manual_seed(0)
>>> watts = torch.cat([
...     torch.zeros(60),                    # idle
...     torch.full((60,), 2000.0),          # a kettle: on at 60, off at 120
...     torch.zeros(40),                    # idle
...     torch.linspace(0.0, 1500.0, 60),    # something ramping, from 160
...     torch.full((40,), 1500.0),          # and holding
... ]) + torch.randn(260) * 4.0
>>> detector = nn_.ZScoreDetector(window=12, threshold=3.0, min_gap=20, min_delta=100.0)
>>> mask = detector(watts)
>>> tuple(mask.shape), mask.dtype
((260,), torch.bool)
>>> mask.nonzero().flatten().tolist()
[60, 120]
```

That signal is used for every example on this page, so the detectors can be read against
each other rather than each against its own toy.

The output is a mask rather than a list of indices because a list's length depends on
the *contents* of the input, so a batch of them goes ragged and cannot be stacked. A
mask has the input's shape:

```{doctest}
>>> tuple(detector(torch.stack([watts, watts])).shape)
(2, 260)
```

To get spans instead, convert:

```{doctest}
>>> nn_.segments_from_mask(mask, min_length=10)
[(0, 60), (60, 120), (120, 260)]
```

Three segments, cut at the two events. `min_length` drops slivers, which matters because
a switching transient can flag several adjacent samples and produce a run of two-sample
segments nobody wants.

(detector-catalogue)=
## The detectors

In alphabetical order, because there is no ranking that survives a change of corpus —
{ref}`the benchmark <event-benchmark>` below shows the order inverting between two
houses. Each entry names the paper it comes from, what question it asks, its parameters,
and when it is the one to reach for. New detectors are added here as they are
implemented; nothing above or below this section changes when they are.

### ActiveSectionDetector

*Wild et al. (2015).* Asks which stretches of the signal are **not** steady, and returns
those intervals rather than instants.

```{doctest}
>>> active = nn_.ActiveSectionDetector(window=8, threshold=40.0, min_steady=4)
>>> active.sections(watts)
[(60, 67), (120, 127), (165, 222)]
```

The two kettle edges come back as short intervals and the ramp as one long one — which
is the point. A washing machine is minutes of varying draw, and asking an edge detector
for its "event time" gets an arbitrary sample inside the ramp, or a dozen events for one
activation.

`window`
: span of the running deviation that decides whether a sample is steady.

`threshold`
: deviation above which a sample counts as active, in the signal's own units.

`min_steady`
: a steady run shorter than this is absorbed into the activity around it, so a momentary
  plateau mid-ramp does not cut one activation into three.

Reach for it when the loads have ramps rather than edges, and when what you want next is
a span to extract features between.

### AdaptiveThresholdDetector

*Jin et al. (2011).* Asks whether the step across a point clears a bar set by the local
noise — `base + scale × σ` — rather than a constant.

```{doctest}
>>> adaptive = nn_.AdaptiveThresholdDetector(window=8, base=50.0, scale=6.0, min_gap=20)
>>> adaptive(watts).nonzero().flatten().tolist()
[52, 112, 158]
```

A fixed threshold is a statement about one house: tuned on a home idling at 300 W it
fires constantly on one idling at 3 kW, because the mains noise floor scales with
whatever is already running. This is the detector that does not inherit that problem.

`window`
: span of the windows either side of the point.

`base`
: the floor of the bar, in the signal's units — what a step must clear on a perfectly
  quiet channel.

`scale`
: how many local standard deviations are added to `base`.

`min_gap`
: suppresses further events within this many samples of one already flagged.

Reach for it when the same configuration has to run across houses, or across a day whose
baseline moves a lot.

### CusumDetector

*Page (1954); the CUSUM arm of Anderson et al. (2012).* Accumulates signed deviation
from a running mean and fires when either accumulator crosses `threshold`, then resets.

```{doctest}
>>> cusum = nn_.CusumDetector(window=16, threshold=3000.0, drift=25.0)
>>> cusum(watts).nonzero().flatten().tolist()
[61, 63, 66, 70, 121, 123, 126, 130, 184, 203, 222]
```

Read that carefully: each kettle edge is reported four times, and the ramp — which no
other edge detector here sees at these settings — three more. Because it *integrates*,
CUSUM responds to a change too gradual for any single sample to look anomalous, and
keeps responding while the change continues. That is both its reason to exist and its
cost.

`window`
: span of the running mean the deviation is measured against.

`threshold`
: accumulated deviation at which it fires.

`drift`
: dead band subtracted from each deviation, so noise does not integrate into a false
  alarm over a long quiet stretch. Set it to roughly the channel's noise level; it is
  the parameter that makes CUSUM usable on real data.

Reach for it for anything with a ramp — a heating element, a compressor spinning up —
and pair it with `min_length` in `segments_from_mask` if you want one event per edge
rather than the whole edge.

### GLRDetector

*The strongest of the three Anderson et al. (2012) compare.* Asks whether one mean
explains the windows either side of a point or two explain them better — a question
about a *step*, which is what switching is.

```{doctest}
>>> glr = nn_.GLRDetector(pre=8, post=8, threshold=100.0, min_gap=20, min_delta=100.0)
>>> glr(watts).nonzero().flatten().tolist()
[59, 119]
```

Two events, one per edge, and a single noise spike barely moves the statistic. Lowering
the threshold brings the ramp in as a series of steps:

```{doctest}
>>> nn_.GLRDetector(pre=8, post=8, threshold=50.0, min_gap=20,
...                 min_delta=100.0)(watts).nonzero().flatten().tolist()
[58, 118, 162, 182, 202]
```

`pre`, `post`
: spans of the windows before and after the candidate point. They need not be equal —
  a short `post` detects sooner, a long one is surer.

`threshold`
: the likelihood-ratio statistic at which it fires.

`min_gap`
: suppresses further events within this many samples of one already flagged.

`min_delta`
: absolute floor on the step, in the signal's units. This is what stops sensor noise on
  an idle channel registering as activity — an idle channel has a tiny deviation, so
  *any* wobble clears a purely relative test.

Reach for it first on an unfamiliar corpus. It is at or near the top on both houses in
the benchmark below, which is what the literature reports too.

### GoodnessOfFitDetector

*The chi-squared arm of Anderson et al. (2012).* Tests the window after a point against
the model fitted to the window before it.

```{doctest}
>>> gof = nn_.GoodnessOfFitDetector(pre=8, post=8, threshold=500.0, min_gap=20,
...                                 min_delta=100.0)
>>> gof(watts).nonzero().flatten().tolist()
[52, 112, 160]
```

Unlike the GLR it sums standardised squared residuals, so it also fires when the level
holds and the *variability* changes — a fan stepping between speeds, a motor beginning
to hunt. Here that sensitivity is what catches the start of the ramp at 160, which the
GLR needed a lower threshold to see. The cost is that merely bursty noise registers too.

`pre`, `post`
: spans of the windows before and after the candidate point.

`threshold`
: the chi-squared statistic at which it fires.

`min_gap`
: suppresses further events within this many samples of one already flagged.

`min_delta`
: absolute floor on the change in level, in the signal's units.

Reach for it when the events you care about are changes in character rather than in
level, and when you can afford the extra false alarms.

### MultivariateDetector

*Houidi et al. (2019).* Runs another detector across several measured quantities and
combines the verdicts.

```{doctest}
>>> _ = torch.manual_seed(1)
>>> vars_ = torch.cat([torch.zeros(58), torch.full((62,), 700.0),
...                    torch.zeros(140)]) + torch.randn(260) * 2.0
>>> channels = torch.stack([watts, vars_])
>>> multivariate = nn_.MultivariateDetector(
...     nn_.GLRDetector(pre=8, post=8, threshold=100.0, min_gap=20, min_delta=50.0),
...     rule="vote", align=8)
>>> multivariate(channels).nonzero().flatten().tolist()
[51, 111]
```

A power-factor-correcting load can switch with almost no change in active power while
reactive power steps hard, so detecting on `p` alone throws information away — and a
store here already carries `v`, `i` and `p` on one clock. Input is `(C, T)` or
`(B, C, T)`; output is one verdict per sample, not one per channel.

`detector`
: the per-channel detector. Shared, not copied, so its parameters are the same on every
  channel.

`rule`
: `"any"` fires when any channel does — most sensitive, most false alarms. `"all"`
  demands unanimity. `"vote"` fires on a majority, and is usually the right one.

`align`
: samples within which two channels' events count as the same event. Channels do not
  fire on the same sample — a reactive step leads an active one through the meter's own
  filtering — and without this, `"all"` and `"vote"` would demand a coincidence the
  instruments never produce.

Reach for it whenever you have more than active power. It costs one forward pass per
channel and is the cheapest accuracy on this page.

### ZScoreDetector

Asks whether a sample deviates from its local mean by more than `threshold` running
standard deviations.

```{doctest}
>>> zscore = nn_.ZScoreDetector(window=12, threshold=3.0, min_gap=20, min_delta=100.0)
>>> zscore(watts).nonzero().flatten().tolist()
[60, 120]
```

Both edges, exactly on the sample, and the ramp ignored — a gradual change never makes
any individual sample anomalous. That is the trade in one line.

`window`
: span of the running statistic. **The running deviation is computed over a window that
  includes the event**, so a change has to be sustained relative to the window length to
  score against it: a single-sample spike inflates its own window's variance and hides
  in it, and so does a step shorter than the window. Set `window` shorter than the
  shortest plateau you care about.

`threshold`
: in units of the running deviation. Because the denominator moves with the signal this
  is scale-free — it means the same thing on a 30 W laptop and a 3 kW kettle.

`min_gap`
: suppresses further events within this many samples of one already flagged. Without it
  a single switching transient fires on every sample of its own edge and one event gets
  reported as a dozen.

`min_delta`
: absolute floor, in the signal's units. 100 W is a reasonable start for whole-house
  data.

Reach for it for switching loads — kettles, toasters, lights — where the transition is
one sample wide at any sensible rate and the level change is large. It is also the
cheapest thing here, which matters when you are sweeping a year of data.

## Where a detector marks the event

The examples above put the same kettle switch-on at 60, 59, 52 and 51. None of them is
wrong. A detector comparing a window before a point with a window after it fires as soon
as its *post* window touches the change, so its mark leads the true instant by up to
`post` samples; `MultivariateDetector` leads by a further `align` while it reconciles
channels; and `ZScoreDetector`, whose statistic is computed at the sample itself, lands
on it.

This is why nothing downstream compares timestamps for equality.
{func}`~nilmframe.eval.match_events` pairs a detection with a true event when it lands
within `tolerance` samples, and the metrics below are all built on that. Pick a
tolerance from your sampling rate and the width of the windows you configured, not from
optimism.

## Choosing one

The catalogue entries each end with the case they are for. Two things worth saying
across all of them:

**Start with GLR**, unless you already know the loads ramp — in which case start with
`ActiveSection` — or you already have more than active power, in which case wrap either
in `Multivariate`.

**Then tune the threshold against your own data, not against a default.** Every number
in this page's examples was chosen for one 260-sample toy. De Baets et al. (2017) is
worth reading on this: a comparison at thresholds somebody guessed is a comparison of
the guesses.

(event-benchmark)=
## What they score on real data

```{image} ../_static/datasets/detectors-ukdale-light.png
:class: only-light
:alt: The detectors on UK-DALE house 1: aggregate with true and detected events, F1 per detector, and precision against recall.
```

```{image} ../_static/datasets/detectors-ukdale-dark.png
:class: only-dark
:alt: The detectors on UK-DALE house 1: aggregate with true and detected events, F1 per detector, and precision against recall.
```

The ground truth is built the way the literature builds it: nobody hand-marked switching
instants in these corpora, but there is a submeter per appliance, so a true event is a
*submeter* crossing its on/off threshold and the detectors only ever see the
*aggregate*. Each detector's threshold is swept on the first half of the day and scored
on the second.

One day, tuned on the first half, reported on the second:

```{list-table}
:header-rows: 1
:widths: 22 13 13 13 13 13 13

* - Detector
  - UK-DALE F1
  - precision
  - recall
  - REFIT F1
  - precision
  - recall
* - GLR
  - **0.64**
  - 0.55
  - 0.77
  - 0.54
  - 0.42
  - 0.77
* - Multivariate
  - 0.64
  - 0.59
  - 0.71
  - 0.57
  - 0.44
  - 0.80
* - ActiveSection
  - 0.21
  - 0.21
  - 0.21
  - **0.58**
  - 0.59
  - 0.57
* - ZScore
  - 0.43
  - 0.29
  - 0.82
  - 0.53
  - 0.39
  - 0.80
* - Adaptive
  - 0.38
  - 0.28
  - 0.57
  - 0.51
  - 0.39
  - 0.73
* - GoodnessOfFit
  - 0.38
  - 0.28
  - 0.57
  - 0.51
  - 0.38
  - 0.77
* - CUSUM
  - 0.14
  - 0.29
  - 0.09
  - 0.12
  - 0.08
  - 0.30
```

Three things to read off it.

**Precision is the binding constraint, not recall.** Every detector recalls 0.6–0.8 of
the true events and none gets past 0.6 precision. That is mostly not the detectors'
fault: UK-DALE house 1 has five submeters and far more than five appliances, so
anything unmetered that switches is a real transition in the aggregate with no
ground-truth event behind it, scored a false positive. A number here is a lower bound.

**The ranking is not stable across houses.** `ActiveSection` is last on UK-DALE and
first on REFIT. UK-DALE house 1's submetered loads are mostly edges, REFIT house 1's
include more ramping ones, and a detector that answers in intervals wins exactly when
the events are intervals. Pick against your corpus, not against a table.

**F1 hides which failure you have**, which is why the third panel plots precision
against recall and {class}`~nilmframe.eval.EventCounts` reports the raw tallies.
`ZScore` and `CUSUM` on UK-DALE score 0.43 and 0.14 — but z-score finds 36 of 44 events
and invents 89, while CUSUM finds 4 and invents 10. Those are opposite problems and
they are not fixed the same way.

Reproduce with `docs/_scripts/benchmark_detectors.py` once the subsets are fetched.

## Scoring a detector

{doc}`evaluation`'s {class}`~nilmframe.eval.DetectionF1` answers *which appliance is on
in this window*. Event detection does not know about appliances, so it needs its own
metrics:

```{doctest}
>>> from nilmframe.eval import EventF1, EventTimingError
>>> predicted = torch.tensor([10, 51, 200])
>>> actual = torch.tensor([12, 50])
>>> f1 = EventF1(tolerance=5)
>>> _ = f1.update(predicted, actual)
>>> round(float(f1.compute()), 3)
0.8
>>> timing = EventTimingError(tolerance=5)
>>> _ = timing.update(predicted, actual)
>>> float(timing.compute())
1.5
```

A detection counts when it lands within `tolerance` samples of a true event, and each
true event can be claimed once — without that second rule a detector that fires on
every sample scores perfect recall. {func}`~nilmframe.eval.match_events` exposes the
pairing if you want to inspect it.

`EventTimingError` matters separately from F1: a detector can score well at a loose
tolerance and still be useless for extracting the transient, and this is the number
that says so.

## On a measurement

{class}`~nilmframe.measurement.Measurement` wraps detection and cutting, so exploring
does not require assembling a detector by hand:

```{doctest}
>>> pulse = torch.cat([torch.zeros(40), torch.full((40,), 2000.0), torch.zeros(40)])
>>> m = nf.Measurement.from_power(pulse, fs=1.0)
>>> int(m.events(window=8, threshold=2.5, min_delta=100.0).sum())
2
```

Two events — on and off. And {meth}`~nilmframe.measurement.Measurement.segments` cuts
the measurement at them, returning measurements you can carry on working with:

```{doctest}
>>> [s.n_samples for s in m.segments(window=8, threshold=2.5, min_delta=100.0)]
[40, 40, 40]
```

Off, on, off. Each segment is a full measurement — plot it, take its power, feed it to
a model.

Detection is only meaningful *before* alignment. Aligned data has already been cut at
cycle boundaries and reordered onto a fixed grid; the sample index no longer maps to
wall-clock time in the way an event detector assumes.

## Building activations from an unlabelled recording

The practical use. Detect, cut, keep the segments that are on, and write them as
activations:

```{code-block} python
m = store.measurement("house_1-mains")
mask = nn_.CusumDetector(window=64, threshold=200.0, drift=5.0)(m.p)
spans = nn_.segments_from_mask(mask, min_length=60)

on = [(a, b) for a, b in spans if float(m.p[a:b].mean()) > 100.0]
```

That gives you `[start, stop)` intervals to hand to
{class}`~nilmframe.store.StoreWriter` as {class}`~nilmframe.store.Activation`
records. From there the store knows which windows contain what, and
{doc}`datasets` can build presence targets — see {doc}`data_loading`.
