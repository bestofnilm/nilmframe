# Getting started

Fifteen minutes, no download. {func}`nilmframe.example_store` generates a small
two-house corpus on first use, and every snippet below runs against it.

For the reasoning behind any of this, the {doc}`user guide <user_guide/index>` covers
each step properly. This page is the tour.

## Install

```{code-block} bash
pip install -e ".[all]"     # everything
pip install -e .            # light core: torch, numpy, pandas, pyarrow
```

There is no compiled extension. Cycle alignment is pure PyTorch, so nothing here needs
a compiler on the target machine.

## 1. Look at a measurement

{class}`~nilmframe.measurement.Measurement` is the object for exploring: dot into it,
chain on it, plot it.

```{doctest}
>>> import nilmframe as nf
>>> m = nf.example_measurement("kettle")
>>> m
Measurement(waveform raw, 6000Hz, 0.500s, kettle, 2926W)
>>> round(float(m.vrms), 1), round(float(m.power_factor()), 3)
(230.0, 1.0)
```

Power factor 1.0, because a kettle is a resistor. Chain to align it and its harmonic
content becomes available:

```{doctest}
>>> m.aligned(cycle_size=128)
Measurement(waveform 23x128, 6000Hz, 0.491s, kettle, 2925W)
>>> m.aligned(cycle_size=128).harmonics(4)
tensor([7.5836e-09, 1.0000e+00, 1.9986e-02, 9.9818e-03])
```

→ {doc}`user_guide/measurements`

## 2. Open a store

A store is tabular metadata plus memory-mapped signals. Nothing is loaded until a window
is asked for.

```{doctest}
>>> store = nf.example_store()
>>> store
Store('nilmframe-example-store', channels=8, appliances=4, datasets=['example'])
>>> store.describe()
   appliance  channels  instances  brands     hours
0     fridge         2          2       2  0.001111
1     kettle         2          2       2  0.001111
2     laptop         1          1       1  0.000556
3  microwave         1          1       1  0.000556
```

Your own data goes in through a reader — `nilmframe convert ukdale --src ... --dst ...`
on the command line, or a generator yielding {class}`~nilmframe.store.Recording`
objects.

→ {doc}`user_guide/data_loading`

## 3. Choose a rate

A *view* decides how stored samples become model input. This is the only thing that
changes between a 1 Hz experiment and a 16 kHz one:

```{doctest}
>>> view = nf.HighFreqView(n_cycles=10, cycle_size=64, align="fitps")
>>> view.window_samples(6000.0, 50.0)
1716
```

Windows are specified in **cycles**, not samples, so alignment yields a fixed shape no
matter what the mains frequency did.

→ {doc}`user_guide/views`, {doc}`user_guide/alignment`

## 4. Build a dataset

```{doctest}
>>> from nilmframe.data import WindowDataset, collate_windows
>>> ds = WindowDataset(store, store.submeters().channel_id.tolist(), view=view)
>>> len(ds)
36
>>> item = ds[0]
>>> tuple(item["i"].shape), tuple(item["presence"].shape)
((10, 64), (4,))
```

Each item carries three targets, deliberately not one vector: `presence` (is it on),
`power` (how many watts), and `power_mask` (is that number *known*). `p_total` is the
aggregate **measured from the input signal**, never the sum of labels.

It is an ordinary `torch.utils.data.Dataset`:

```{doctest}
>>> import torch
>>> loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=collate_windows)
>>> tuple(next(iter(loader))["i"].shape)
(8, 10, 64)
```

→ {doc}`user_guide/datasets`

## 5. Split without leaking

A split is an experimental claim, so each protocol is a named object carrying a
manifest, and {func}`~nilmframe.data.check_leakage` proves the claim holds.

```{doctest}
>>> from nilmframe.data import check_leakage
>>> split = nf.LeaveHouseOut(test_size=0.5, seed=0).apply(store)
>>> split.manifest["groups"]
{'train': ['house_2'], 'val': ['house_1'], 'test': []}
>>> check_leakage(split, store, keys=("house", "instance_id"))
[]
```

Trained on house 2, evaluated on house 1, and nothing leaked. That line belongs in a
paper.

→ {doc}`user_guide/splitting`

## 6. Reach for a model

Eleven published architectures, one call signature. Nothing is trained here — the
weights are random — but the shape of the contract is the whole point:

```{doctest}
>>> import nilmframe.nn as nn_
>>> model = nn_.models.build("seq2point", n_appliances=store.n_appliances, window=99)
>>> out = model(torch.rand(4, 1, 99) * 500)
>>> model.kind, tuple(out.shape)
('seq2point', (4, 4, 1))
```

Swap the name and nothing around it changes:

```{doctest}
>>> other = nn_.models.build("unet", n_appliances=store.n_appliances, window=64, depth=2)
>>> other.kind, tuple(other(torch.rand(4, 1, 64) * 500).shape)
('seq2seq', (4, 4, 64))
```

→ {doc}`user_guide/models`

## 7. Score it

```{doctest}
>>> from nilmframe.eval import MeanAbsoluteError
>>> metric = MeanAbsoluteError()
>>> _ = metric.update(torch.zeros(1, 4), torch.zeros(1, 4))
>>> float(metric.compute())
0.0
```

→ {doc}`user_guide/evaluation`

## From the command line

```{code-block} bash
nilmframe convert ukdale --src .../low_freq --dst ~/.nilm/ukdale --rate-hz 0.1667
nilmframe describe ~/.nilm/ukdale --verify
nilmframe compat   ~/.nilm/ukdale
```

→ {doc}`user_guide/cli`

## Where next

Read the {doc}`user guide <user_guide/index>` in order if you are starting an
experiment, or jump to the chapter you need. {doc}`concepts` is the short version of
why the design is the way it is.
