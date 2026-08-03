# Datasets and loaders

{class}`~nilmframe.data.WindowDataset` is a `torch.utils.data.Dataset`. It behaves
exactly the way a PyTorch dataset is expected to behave — indexable, sized, safe to
hand to a `DataLoader` with workers — and it composes with the rest of PyTorch rather
than replacing any of it.

```{doctest}
>>> import torch, nilmframe as nf
>>> from nilmframe.data import WindowDataset, collate_windows
>>> store = nf.example_store()
>>> ids = store.submeters().channel_id.tolist()
>>> view = nf.HighFreqView(n_cycles=10, cycle_size=64, align="fitps")
>>> ds = WindowDataset(store, ids, view=view)
>>> len(ds)
36
```

Three things go in: the store, the list of channel ids this dataset covers — usually
one fold of a {doc}`split <splitting>` — and the {doc}`view <views>`. The dataset
computes a window index from those and nothing else; it holds no signal data.

## What an item contains

```{doctest}
>>> item = ds[0]
>>> sorted(item)
['channel', 'cycle_mask', 'i', 'p_total', 'power', 'power_mask', 'presence', 'start', 'v']
```

They split into three groups.

**Signals** — `v`, `i`, `cycle_mask`, `p_total`. These come from the view, and these
are the only keys a model is permitted to read.

```{doctest}
>>> tuple(item["v"].shape), item["cycle_mask"].dtype
((10, 64), torch.bool)
>>> round(float(item["p_total"]), 1)
2924.8
```

**Targets** — `presence`, `power`, `power_mask`. Deliberately three tensors rather
than one:

```{doctest}
>>> item["presence"], item["power_mask"]
(tensor([0., 1., 0., 0.]), tensor([True, True, True, True]))
```

`presence` is *is it on*. `power` is *how many watts*. `power_mask` is *is that number
actually known* — and it is the one that is usually missing from NILM pipelines and
usually matters. An aggregate recording annotated only with on/off times knows presence
but not per-appliance power. Training a regression head against an invented zero there
teaches the model that every annotated appliance draws nothing, which is worse than
not training it at all.

Here is the same window from a mains channel, where no submeter truth exists:

```{doctest}
>>> mains = WindowDataset(store, store.mains().channel_id.tolist(), view=view)
>>> mains[0]["power_mask"]
tensor([False, False, False, False])
```

Every entry false. The loss will skip the power term for this window entirely rather
than fitting to a fiction.

**Provenance** — `channel` and `start`. Which channel and which sample offset this
window came from, so any prediction can be traced back to the signal that produced it.

```{doctest}
>>> item["channel"], item["start"]
('house_1-kettle', 0)
```

## Collating

`collate_windows` is the batching function. Pass it to `DataLoader`:

```{doctest}
>>> loader = torch.utils.data.DataLoader(
...     ds, batch_size=8, shuffle=True, collate_fn=collate_windows)
>>> batch = next(iter(loader))
>>> tuple(batch["i"].shape), tuple(batch["presence"].shape)
((8, 10, 64), (8, 4))
```

It stacks tensors, leaves the string columns as lists, and — importantly — raises a
legible error when items disagree about which keys they have, rather than producing a
batch that is quietly missing a target:

```{doctest}
>>> a, b = ds[0], ds[1]
>>> del b["power"]
>>> collate_windows([a, b])
Traceback (most recent call last):
    ...
ValueError: items disagree on keys: {'power'} present in some but not all
```

That check exists because the failure it catches used to appear three layers later,
as a shape error inside a loss function.

## Inspecting

Two methods help when a batch is doing something you did not expect.

{meth}`~nilmframe.data.WindowDataset.describe` reports what the dataset is, including
the view's settings, so an experiment log records how its inputs were built:

```{doctest}
>>> ds.describe()
{'windows': 36, 'channels': 6, 'appliances': ['fridge', 'kettle', 'laptop', 'microwave'], 'targets': ['presence', 'power'], 'unknown_appliances': [], 'view': 'highfreq', 'n_cycles': 10, 'cycle_size': 64, 'align': 'fitps'}
```

{meth}`~nilmframe.data.WindowDataset.measurement` gives you window `i` as a
{doc}`Measurement <measurements>`, so you can look at it with the same tools you use
for exploring:

```{doctest}
>>> ds.measurement(0)
Measurement(waveform 10x64, 6000Hz, 0.107s, 2925W)
```

That is the debugging loop: a batch looks wrong, you pull the offending index out as a
measurement, plot it, and see that the window landed on a gap.

## Options

`stride` controls window overlap, as a fraction of window length. `1.0` is
non-overlapping; `0.5` steps half a window at a time and roughly doubles the count.

`targets` selects which target tensors to build. Drop `power` if you are only doing
detection and the regression head is not in the model.

`max_windows_per_channel` caps the contribution of any one channel. Useful when one
house is ten times longer than the others and would otherwise dominate the epoch.

`augment` runs after the targets are built — this is where synthetic aggregation and
jitter go; see {doc}`augmentation`.

## Epochs and randomness

Transforms that draw randomness need to draw *differently each epoch* but
*reproducibly*. Call {meth}`~nilmframe.data.WindowDataset.set_epoch` at the top of
each epoch:

```{code-block} python
for epoch in range(n_epochs):
    train.set_epoch(epoch)
    for batch in loader:
        ...
```

The seed for window `i` in epoch `e` is a function of both, so a run is reproducible
end to end, and worker processes do not all draw the same augmentation — the standard
`DataLoader` failure where every worker forks the same RNG state.

## Materialising

Lazy reading is the right default, but sometimes you want the windows on disk: an
ablation you will run twenty times, or a cluster job where the store lives behind a
slow mount. {func}`~nilmframe.data.materialize` writes them out:

```{code-block} python
nf.materialize(train, "cache/train.pt", n_samples=50_000, seed=0)
```

This is an optimisation, not a different pipeline — the windows are the same ones
`ds[i]` would have produced.
