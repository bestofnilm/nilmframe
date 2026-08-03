# Splitting data

Most of the difference between a NILM number that means something and one that does
not is in the split. It is easy to report 0.95 F1 on a protocol that tested a model on
the same physical kettle it was trained on, and nothing about the training code will
tell you that is what happened.

So a split here is not a pair of index arrays. It is a named protocol object that
produces a {class}`~nilmframe.data.Split` carrying a **manifest** — what protocol,
what seed, which groups landed on which side — and a checker that proves the claim.

```{doctest}
>>> import nilmframe as nf
>>> store = nf.example_store()
>>> split = nf.LeaveHouseOut(test_size=0.5, seed=0).apply(store)
>>> split
Split(train=4, val=4, test=0, protocol=LeaveHouseOut)
>>> split.manifest["groups"]
{'train': ['house_2'], 'val': ['house_1'], 'test': []}
```

You can read off exactly what was tested: trained on house 2, evaluated on house 1.
That line belongs in a paper.

## The protocols

**{class}`~nilmframe.data.RandomSplit`** groups by *instance* — a physical unit, not
an appliance class — and splits those groups. So the same kettle never appears on both
sides, even though kettles do:

```{doctest}
>>> rs = nf.RandomSplit(test_size=0.3, seed=0).apply(store)
>>> rs
Split(train=6, val=2, test=0, protocol=RandomSplit)
```

This is the weakest honest protocol. It answers "does the model recognise appliances it
has seen examples of", which is a real question, but not the one most papers claim to
be answering.

**{class}`~nilmframe.data.LeaveHouseOut`** holds out whole houses. This is the
protocol that tests deployment: a new installation, with its own wiring, its own
supply impedance, its own mix of loads.

**{class}`~nilmframe.data.LeaveBrandOut`** holds out manufacturers:

```{doctest}
>>> nf.LeaveBrandOut(test_size=0.5, seed=0).apply(store).manifest["groups"]
{'train': ['globex'], 'val': ['acme'], 'test': []}
```

This is the hardest of the three and the one that most often collapses a headline
number. Two fridges from the same manufacturer share a compressor and a control board,
and a model that has learned that specific signature has not learned "fridge".

**{class}`~nilmframe.data.CrossDataset`** trains on one corpus and tests on another,
by name. Different instrumentation, different country, different everything:

```{code-block} python
split = nf.CrossDataset(train_on=["ukdale"], test_on=["plaid"]).apply(merged)
```

It refuses to build a split where a dataset appears on both sides, rather than
silently intersecting them.

All four take `test_size`, `seed`, and an optional `holdout_size` that carves out a
third fold. The third fold is empty unless you ask for it, because a test set you
looked at during development is a validation set.

## Proving there is no leakage

{func}`~nilmframe.data.check_leakage` returns a list of violations. Empty means the
claim holds:

```{doctest}
>>> from nilmframe.data import check_leakage
>>> check_leakage(split, store)
[]
```

By default it checks `instance_id`. Pass whatever the protocol claims to separate:

```{doctest}
>>> check_leakage(split, store, keys=("house", "instance_id"))
[]
```

And it does find real violations. `RandomSplit` never claimed to separate houses, and
it does not:

```{doctest}
>>> check_leakage(rs, store, keys=("house",))
["house='house_1' appears in both 'train' and 'val'"]
```

That is not a bug in `RandomSplit` — it is the check doing its job. Run it in CI on
whatever protocol your paper claims, so the claim cannot rot while the code changes
around it.

## Looking at what you got

{meth}`~nilmframe.data.Split.summary` shows the shape of each fold:

```{doctest}
>>> split.summary(store)
    fold  channels  instances  appliances     hours
0  train         4          4           3  0.002222
1    val         4          4           3  0.002222
2   test         0          0           0  0.000000
```

The column to watch is `appliances`. If a class is present in train but absent from
val, the metric for that class is undefined, and depending on how the metric averages
it will either be silently dropped or silently counted as zero. Neither is what you
want to discover after the fact.

## Open set

Real deployments meet appliances that were not in the training vocabulary. The usual
handling — a "other" class trained on whatever happened to be left over — teaches the
model that "unknown" has a signature, which it does not.

{class}`~nilmframe.data.UnseenAppliance` handles it structurally instead. Held-out
classes get **no column at all** in the label space:

```{doctest}
>>> openset = nf.UnseenAppliance(unknown=["microwave"], seed=0).apply(store)
>>> openset.manifest["known"], openset.manifest["unknown"]
(['fridge', 'kettle', 'laptop'], ['microwave'])
```

The model has nowhere to put a microwave, so the only way to be right about a window
containing one is to say "not mine". Those windows are flagged `is_unknown`, contribute
no presence target and no attributable power, and are scored by
{class}`~nilmframe.eval.UnknownAUROC` rather than by the per-class metrics.

## Using a split

The folds are channel-id lists, so they go straight into a dataset:

```{doctest}
>>> view = nf.HighFreqView(n_cycles=5, cycle_size=64)
>>> train = nf.WindowDataset(store, split.train, view=view)
>>> val = nf.WindowDataset(store, split.val, view=view)
>>> len(train), len(val)
(48, 48)
```

Build the split **once** and share it across every arm of a comparison. Rebuilding it
per arm lets the seed drift between them, which quietly turns a controlled comparison
into several unrelated experiments.
