# Augmentation

The scarcity in NILM is not signal, it is *labelled aggregates*. Submetered
recordings of individual appliances are plentiful; recordings of a whole house where
somebody wrote down exactly which appliances were on and how much each drew are rare,
short, and mostly not public.

The standard answer is to build synthetic aggregates by superposing submetered
recordings. The standard mistake is to do it once, offline, and write the result to
disk as a fixed dataset. That fixes the mixtures — the model sees the same ten thousand
combinations every epoch and overfits them — and it hides the construction inside a
preprocessing script nobody reruns.

Here mixing is an *augmentation*, applied per item, per epoch, on the fly.

## Synthetic aggregation

{class}`~nilmframe.data.MixAggregate` superposes several submetered windows into one:

```{doctest}
>>> import torch, nilmframe as nf
>>> from nilmframe.data import WindowDataset, MixAggregate
>>> store = nf.example_store()
>>> ids = store.submeters().channel_id.tolist()
>>> view = nf.HighFreqView(n_cycles=8, cycle_size=64)
>>> mixed = WindowDataset(store, ids, view=view,
...                       augment=MixAggregate(k=(1, 3), p=1.0), seed=0)
>>> item = mixed[0]
>>> item["n_components"]
2
>>> item["presence"]
tensor([1., 1., 0., 0.])
```

Two appliances went in, and both show up in `presence`. Compare with the unaugmented
dataset, where a submeter window contains exactly one appliance:

```{doctest}
>>> plain = WindowDataset(store, ids, view=view)
>>> plain[0]["presence"]
tensor([0., 1., 0., 0.])
```

The mixture's targets are *exact*, not estimated, because the components are known:

```{doctest}
>>> bool(torch.allclose(item["power"].sum(), item["p_total"], rtol=1e-5))
True
```

The per-appliance powers sum to the measured aggregate to within float tolerance. That
is a stronger guarantee than any real aggregate recording offers, and it is what makes
synthetic mixtures worth training on: a conservation term has
something exactly right to learn from.

Superposition is on **waveforms**, not on powers. Two loads with different phase
relationships to the voltage partly cancel, so their combined current is not the sum of
their RMS currents — and their combined active power is not always the sum of their
individual active powers in the way a scalar model would predict. Mixing at the
waveform level gets this right for free.

### Choosing k

`k` is the inclusive range of components per mixture:

```{doctest}
>>> MixAggregate(k=(1, 4), p=1.0)
MixAggregate(k=(1, 4), p=1.0)
```

Start the range at 1. A model trained only on mixtures of three or more never sees
what a single appliance looks like on its own, and then does badly on exactly the
windows that should be easiest. `p` is the probability of mixing a given window at
all, which is another way to keep unmixed examples in the distribution.

`same_rate_only` defaults to true and should stay that way. Superposition assumes the
two windows share a supply voltage. Two recordings made on different rigs at different
rates do not, and mixing them produces a signature no meter would ever see.

### It varies per epoch

Call {meth}`~nilmframe.data.WindowDataset.set_epoch` and the same index yields a
different mixture:

```{doctest}
>>> mixed.set_epoch(1)
>>> mixed[0]["n_components"]
3
```

Reproducibly different: the seed for window `i` in epoch `e` is a function of both, so
a run repeats exactly while still showing the model fresh combinations. This is also
what keeps `DataLoader` workers from all drawing the same augmentation, which is the
usual way a forked RNG quietly halves the diversity of an epoch.

## Jitter

Two smaller augmentations model measurement conditions rather than load combinations.

{class}`~nilmframe.data.VoltageJitter` perturbs the supply level. Real supply voltage
is not 230 V; it moves with load and time of day, and an appliance's current moves with
it.

{class}`~nilmframe.data.GainJitter` perturbs the measurement gain, modelling the fact
that two current transformers are not identically calibrated. This is the one that
matters for cross-dataset generalisation, because a model can otherwise learn a
corpus's specific calibration error as if it were a property of the appliances.

```{doctest}
>>> from nilmframe.data import VoltageJitter, GainJitter
>>> VoltageJitter(sigma=0.02), GainJitter(sigma=0.05)
(VoltageJitter(sigma=0.02), GainJitter(sigma=0.05))
```

Both take a relative standard deviation. 2 % and 5 % are reasonable defaults; larger
values start teaching the model that power is unreliable, which it is not.

## Composing

{class}`~nilmframe.data.Compose` chains them, in order:

```{doctest}
>>> from nilmframe.data import Compose
>>> augment = Compose([MixAggregate(k=(1, 3)), VoltageJitter(0.02), GainJitter(0.05)])
>>> augment
Compose(MixAggregate(k=(1, 3), p=1.0), VoltageJitter(sigma=0.02), GainJitter(sigma=0.05))
>>> ds = WindowDataset(store, ids, view=view, augment=augment, seed=0)
>>> sorted(ds[0])[:4]
['channel', 'cycle_mask', 'i', 'n_components']
```

Mix first, then jitter. The other order jitters each component separately and then
adds them, which models a rig where every appliance had its own miscalibrated meter —
occasionally what you want, usually not.

## What a mixture does to the targets

The combination rules are worth stating explicitly, because they are where a synthetic
aggregate can go quietly wrong:

- **`presence`** combines by **max**. On is on.
- **`power`** combines by **sum**. Each component's watts are known exactly.
- **`power_mask`** combines by **AND**. A mixture is only as knowable as its
  least-known part — mix a submeter window with an aggregate window whose per-appliance
  power is unknown and the whole mixture's power becomes unknown.
- **`p_total`** is recomputed from the summed waveform, not summed from the parts.

That last one is not a detail. `p_total` is the model's input and the conservation
target; taking it from the components rather than from the resulting signal would
reintroduce exactly the label leakage the whole design is built to avoid.

## Evaluating on real aggregates

Augment the training fold only. The validation and test folds should be real mains
windows, because "does it work on synthetic mixtures" is not the question anyone is
asking:

```{code-block} python
train = nf.WindowDataset(store, split.train, view=view,
                         augment=Compose([MixAggregate(k=(1, 4)), GainJitter(0.05)]))
val   = nf.WindowDataset(store, split.val, view=view)          # no augment
```

The gap between the two is itself informative. A model that does well on synthetic
mixtures and badly on real mains has learned something about the superposition, not
about the appliances.
