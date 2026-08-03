# Cycle alignment

Take two consecutive 1200-sample slices of a 6 kHz recording of 50 Hz mains. Each is
nominally ten cycles. Plot them on top of each other and they do not line up: mains
frequency is not exactly 50 Hz, it drifts by a few tens of millihertz over minutes, and
the slice boundaries do not fall on zero crossings anyway.

For a convolutional model this is a problem. The same appliance, sampled twice, presents
the same waveform at a different phase, and the network has to spend capacity learning
that phase does not matter. For a model that looks at cycle shape — which is where the
discriminative information in a high-frequency signature lives — it is worse than
inefficient; it blurs exactly the feature being looked at.

**FITPS** (frequency-invariant transformation of periodic signals) removes it. Locate
the voltage's rising zero crossings, cut the signal at them, and resample each cycle
onto a common grid. Every cycle then starts at the same electrical phase and has the
same number of points, regardless of what the mains was doing.

## Finding the crossings

A rising crossing lies between samples `k` and `k+1` when `v[k] < 0` and `v[k+1] >= 0`.
Linear interpolation puts it at `k + v[k] / (v[k] - v[k+1])` — to sub-sample precision,
which matters, because at 6 kHz one sample is 1.5 electrical degrees:

```{doctest}
>>> import torch, nilmframe as nf
>>> m = nf.example_measurement()
>>> pos, mask = nf.nn.rising_zero_crossings(m.v.unsqueeze(0))
>>> int(mask.sum())
24
>>> [round(float(x), 2) for x in pos[0][mask[0]][:3]]
[120.0, 240.0, 360.0]
```

Twenty-four crossings in half a second, 120 samples apart — exactly 50 Hz at 6 kHz, as
this generated corpus should be. On real data the spacing wanders by a sample or two,
and that wander is the whole reason for doing this.

The mains frequency itself falls out of the crossing spacing:

```{doctest}
>>> nf.nn.estimate_f0(m.v.unsqueeze(0), fs=6000.0)
tensor([50.])
```

## Aligning

{func}`~nilmframe.nn.cycle_align` does the whole thing:

```{doctest}
>>> v, i = m.v.unsqueeze(0), m.i.unsqueeze(0)
>>> vc, ic, cycle_mask = nf.nn.cycle_align(v, i, fs=6000.0, cycle_size=128)
>>> tuple(vc.shape), tuple(cycle_mask.shape)
((1, 23, 128), (1, 23))
>>> int(cycle_mask.sum())
23
```

Twenty-three complete cycles out of twenty-four crossings — the last crossing has no
successor to cut against, so it does not open a cycle. Each is resampled from its
native ~120 points onto the requested 128.

Ask for a fixed count and you get exactly that count:

```{doctest}
>>> vc, ic, cycle_mask = nf.nn.cycle_align(v, i, fs=6000.0, cycle_size=128, n_cycles=10)
>>> tuple(vc.shape), int(cycle_mask.sum())
((1, 10, 128), 10)
```

This is what makes batching work. Windows specified in cycles produce a fixed
`(n_cycles, cycle_size)` shape no matter what the mains frequency did, so ragged
batching never arises anywhere downstream.

### When there are not enough cycles

If a window yields fewer complete cycles than requested — a frequency excursion, a gap,
a window that landed at the end of a recording — the shortfall is padded to keep the
shape, and `cycle_mask` marks the padding false.

This is worth being explicit about, because the obvious implementation is wrong. Padding
by repeating cycle 0 and marking it valid produces duplicated data that the model
trains on as though it were measured. The padding here is masked, and every consumer —
the power computation in {doc}`views` — respects the mask.

### How much signal to ask for

{func}`~nilmframe.nn.samples_for_cycles` gives the window length needed to reliably
yield `n_cycles`:

```{doctest}
>>> nf.nn.samples_for_cycles(20, fs=6000.0, f0=50.0)
3150
```

That is `ceil((20 + 1) × 1.25 × 6000 / 50)`: one spare cycle for the partial ones at
each end, times a slack factor for drift. {class}`~nilmframe.data.HighFreqView` calls
this for you.

## As a layer

{class}`~nilmframe.nn.CycleAlign` is the `nn.Module` form:

```{doctest}
>>> align = nf.nn.CycleAlign(cycle_size=64, n_cycles=8, f0=50.0)
>>> align
CycleAlign(cycle_size=64, n_cycles=8, f0=50.0, tol=0.2)
>>> vc, ic, mask = align(m.v.unsqueeze(0), m.i.unsqueeze(0), fs=6000.0)
>>> tuple(ic.shape)
(1, 8, 64)
```

The implementation is pure PyTorch — there is no compiled extension in this library —
so it is batched, differentiable, and runs on whatever device its inputs are on. That
has three consequences worth naming:

**Alignment can be part of the model.** It does not have to happen in a preprocessing
script whose parameters then live somewhere other than the model that depends on them.
Training and deployment run the same code path.

**It runs on the GPU.** For a 16 kHz corpus this is the difference between alignment
being a bottleneck and being free.

**It installs everywhere.** The reference implementation of FITPS is C++; a Python
package that wraps it needs a compiler on the target machine. This one needs `torch`.

## Does it help?

That is an empirical question, and the library is arranged so you can answer it rather
than assume. The control arm is one keyword:

```{code-block} python
aligned = nf.HighFreqView(n_cycles=20, cycle_size=128, align="fitps")
control = nf.HighFreqView(n_cycles=20, cycle_size=128, align=None)
```

Both consume the same number of input samples — `window_samples` is identical for the
two — and both produce the same output shape. The only difference is whether each cycle
was resampled from its detected boundaries or the span was reshaped at fixed length.

Keeping the sample count identical is deliberate, and it is not what published FITPS
comparisons have always done. If the aligned arm requests more signal (because it needs
slack) and the control arm does not, the aligned model has seen more data, and
"alignment helps" and "more input helps" have been confounded. Holding the window fixed
both arms against one store and one split, so the view is the only thing that differs.

## Tolerance

`tol` rejects a candidate cycle whose length is further than this fraction from the
expected period. The default of 0.2 accepts anything between 40 and 60 Hz on a 50 Hz
supply, which is loose enough for real drift and tight enough to reject a false crossing
caused by noise near the zero point.

Tighten it on a clean lab recording. Loosen it — or drop to the unaligned view — if
`cycle_mask` shows a lot of rejected cycles, which is the symptom of a voltage channel
too noisy to cross cleanly.
