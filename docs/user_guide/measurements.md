# Measurements

{class}`~nilmframe.measurement.Measurement` is the object you explore with. It is
what you get when you point at a channel and ask "what does this look like": a thing
you can dot into, chain on, add together and plot, without setting up a dataset, a
view, a split or a loader.

It is a *lens*, not storage. It holds tensors, it is frozen, and every method returns
a new one — nothing you do to a measurement can disturb the store it came from.

## Getting one

Three ways in. From the built-in corpus, by appliance name:

```{doctest}
>>> import nilmframe as nf
>>> m = nf.example_measurement("kettle")
>>> m
Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
```

From any store, by channel id:

```{doctest}
>>> store = nf.example_store()
>>> store.measurement("house_1-mains")
Measurement(waveform raw, 6000Hz, 2.000s, 3290W)
```

Or from a dataset item, which is the one you will reach for while debugging a batch
that is doing something strange — see {doc}`datasets`.

The repr is deliberately dense: what it is, at what rate, how long, which appliances,
and how many watts. Everything you would otherwise print four attributes to find out.

## Reading quantities off it

RMS voltage and current are attributes; the power quantities are methods, because
they each take optional arguments. All of them mean what an engineer means:

```{doctest}
>>> round(float(m.vrms), 1), round(float(m.irms), 2)
(230.0, 12.73)
>>> round(float(m.active_power()), 1)
2926.1
>>> round(float(m.apparent_power()), 1)
2927.4
>>> round(float(m.power_factor()), 3)
1.0
```

A kettle is a resistor, so its power factor is 1 and its apparent power is its active
power. That is a useful sanity check to keep in mind: if a purely resistive load comes
back with a power factor of 0.6, the calibration is wrong, not the appliance.

Active power is `mean(v * i)` over the window — the integral of instantaneous power.
It is not `vrms * irms`, which is apparent power, and the ratio of the two is exactly
the power factor.

## Chaining

Every transformation returns a new measurement, so operations compose left to right:

```{doctest}
>>> m.seconds(0.0, 0.1)
Measurement(waveform raw, 6000Hz, 0.100s, kettle, 2926W)
>>> m.aligned(cycle_size=128)
Measurement(waveform 23x128, 6000Hz, 0.491s, kettle, 2925W)
>>> m.aligned(cycle_size=128).resample(64)
Measurement(waveform 23x64, 6000Hz, 0.245s, kettle, 2923W)
```

Read the middle line carefully, because it is where the library's central idea shows
up in miniature. `raw` became `23x128`: half a second of 50 Hz mains is 25 cycles, and
{meth}`~nilmframe.measurement.Measurement.aligned` found 23 complete ones, located each
one's rising zero crossing to sub-sample precision, and resampled each onto a common
128-point grid. The duration dropped from 0.500 s to 0.491 s because the partial cycles
at the two ends were discarded rather than padded with something invented.

The active power barely moved — 2926 W to 2925 W — which is the check worth doing: if
alignment changed the power, it aligned to the wrong thing. {doc}`alignment` covers the
algorithm.

Once aligned, the frequency-domain methods become available, because "harmonic" only
means something relative to a fundamental:

```{doctest}
>>> a = m.aligned(cycle_size=128)
>>> a.harmonics(4)
tensor([7.5836e-09, 1.0000e+00, 1.9986e-02, 9.9818e-03])
```

Index 0 is DC and is essentially zero, index 1 is the fundamental and is 1.0 because
the result is normalised to it, and the rest are the distortion. A kettle has almost
none. Try the same on a laptop's switched-mode supply and the third harmonic will be a
substantial fraction of the fundamental.

Calling `harmonics` on an unaligned measurement raises rather than guessing:

```{doctest}
>>> m.lowpass(8)
Traceback (most recent call last):
    ...
ValueError: harmonics are only meaningful on aligned cycles; call .aligned()
```

## Adding measurements together

Two measurements add. The result is the waveform sum — what a meter upstream of both
would have recorded — and it *remembers what went into it*:

```{doctest}
>>> kettle = nf.example_measurement("kettle")
>>> fridge = nf.example_measurement("fridge")
>>> mix = kettle + fridge
>>> mix
Measurement(waveform raw, 6000Hz, 0.500s, 2 components, 3164W)
>>> mix.n_components, mix.appliances
(2, ('kettle', 'fridge'))
>>> mix.components[0]
Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
```

The component axis is always present, even on a single measurement — `n_components`
is 1 rather than absent — so nothing downstream has to special-case a mixture. This
is the same mechanism {class}`~nilmframe.data.MixAggregate` uses to synthesise
aggregates during training, and it is why a synthetic aggregate comes with exact
per-appliance ground truth rather than an estimate. See {doc}`augmentation`.

Note that `3164` is not `2926 + 1240`. Adding waveforms is not adding powers: the two
loads have different phase relationships to the voltage, so their instantaneous
currents partly cancel. Summing scalar powers would have got this wrong, which is one
concrete reason the library keeps waveforms around rather than reducing to power
early.

## Plotting

{meth}`~nilmframe.measurement.Measurement.plot` draws voltage and current against
time, on a matplotlib axis you supply or one it creates:

```{code-block} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
kettle.seconds(0, 0.06).plot(ax=ax[0])
kettle.aligned(cycle_size=128).plot(ax=ax[1])
```

Two other visual methods are worth knowing. {meth}`~nilmframe.measurement.Measurement.vi_image`
renders the V-I trajectory — voltage against current over one cycle, which traces a
loop whose shape is characteristic of the load type. {meth}`~nilmframe.measurement.Measurement.spectrogram`
gives the time-frequency view:

```{doctest}
>>> tuple(kettle.aligned(cycle_size=64).vi_image().shape)
(23, 3, 64, 64)
>>> tuple(store.measurement("house_1-mains").spectrogram().shape)
(129, 188)
```

Both are covered properly in {doc}`representations`, where they appear as
`nn.Module` transforms you can put inside a model rather than one-off pictures.

## Handing it to a model

{meth}`~nilmframe.measurement.Measurement.batch` produces exactly the dict a
a model expects, with a leading batch dimension of one:

```{doctest}
>>> batch = kettle.aligned(cycle_size=64).batch()
>>> sorted(batch)
['cycle_mask', 'i', 'p_total', 'v']
>>> tuple(batch["i"].shape)
(1, 23, 64)
```

Those four keys are the entire input contract. There is no `presence` and no `power`
in it, because those are labels and a model that reads them at inference time cannot
be deployed.

`cycle_mask` marks which cycles are real. When alignment finds fewer complete cycles
than asked for, the shortfall is padded so the tensor keeps a fixed shape, and the mask
is how the model knows to ignore the padding rather than learning from fabricated
cycles.

{meth}`~nilmframe.measurement.Measurement.numpy` is the escape hatch, for when you
want to hand the raw arrays to something else entirely:

```{doctest}
>>> sorted(kettle.numpy())
['f0', 'fs', 'i', 't0', 'v']
```

## If you do not use torch

Torch is a hard dependency — the alignment, the views and the models are all batched
tensor operations and there is no numpy fallback for them. But *knowing* torch is not
a requirement for the parts before that, and the boundary accepts numpy in both
directions.

Going in, anything `torch.as_tensor` understands works — numpy arrays, lists, pandas
columns:

```{doctest}
>>> import numpy as np
>>> watts = np.concatenate([np.zeros(40), np.full(40, 2000.0), np.zeros(40)])
>>> m = nf.Measurement.from_power(watts, fs=1.0, appliances=["kettle"])
>>> m.kind, m.n_samples
('power', 120)
```

Coming out, {meth}`~nilmframe.measurement.Measurement.numpy` gives plain arrays, and
the store reads as numpy in the first place — {meth}`~nilmframe.store.Store.read_window`
returns an `ndarray`, not a tensor:

```{doctest}
>>> type(m.numpy()["p"]).__name__
'ndarray'
>>> type(store.read_window("house_1-kettle", "i", 0, 64)).__name__
'ndarray'
```

The {doc}`event detectors <event_detection>` accept the same inputs, so the whole
load-detect-inspect loop can be written without constructing a tensor by hand:

```{doctest}
>>> import nilmframe.nn as nn_
>>> events = nn_.GLRDetector(pre=8, post=8, threshold=100.0, min_delta=100.0)(watts)
>>> events.nonzero().flatten().numpy()
array([39, 79])
```

What comes back is a tensor. `torch.as_tensor` does not copy a numpy array, so this
convenience costs nothing, and `.numpy()` on the result costs nothing either.

## Devices

{meth}`~nilmframe.measurement.Measurement.to` moves every tensor it holds, so a
measurement follows the same idiom as a module:

```{code-block} python
m = nf.example_measurement("kettle").to("cuda")
```

Alignment is pure PyTorch — there is no compiled extension anywhere in the library —
so `m.to("cuda").aligned(cycle_size=128)` runs the whole alignment on the GPU, batched.
That is why alignment can sit inside a model as a layer rather than having to happen in
a preprocessing script.
