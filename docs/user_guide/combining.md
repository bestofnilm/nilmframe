# Combining datasets

There are not many NILM datasets, and each one is small. Almost any question worth
asking — does this generalise across brands, across houses, across countries — needs
more than one. So the question is not whether to combine corpora but under what
conditions combining them is honest.

The answer is not a single yes or no, because it depends on what you are going to do
with the result. Two recordings at different sampling rates cannot be concatenated
into one raw waveform tensor. But after cycle alignment they are the same shape and
the same phase, so for an aligned model the rate difference has stopped existing. A
120 V corpus and a 230 V corpus, on the other hand, differ in a way alignment does not
touch: the same appliance draws roughly twice the current on 120 V, and no amount of
resampling fixes that.

So compatibility is reported *per intended use*, not as a verdict.

## The compatibility report

{func}`~nilmframe.compat.compatibility` inspects one or more stores and reports each
axis along which they differ, along with which pipelines that difference blocks:

```{doctest}
>>> import nilmframe as nf
>>> store = nf.example_store()
>>> report = nf.compatibility(store)
>>> report
CompatibilityReport(datasets=['example'], channels=8, varying=['appliance vocabulary'])
```

`print` gives the full picture:

```{doctest}
>>> print(report.summary())
CompatibilityReport(datasets=['example'], channels=8, varying=['appliance vocabulary'])
<BLANKLINE>
  highfreq_aligned   ok on all 8 channels
  highfreq_raw       ok on all 8 channels
  lowfreq            ok on all 8 channels
<BLANKLINE>
  varying axes:
    appliance vocabulary 4 distinct: ['fridge', 'kettle', 'laptop', 'microwave'])
```

The three named pipelines are the three ways the data can be consumed:
`highfreq_raw` (fixed-length waveform windows), `highfreq_aligned` (cycle-aligned
windows) and `lowfreq` (a power series). A store can be fine for one and useless for
another.

Programmatically:

```{doctest}
>>> report.is_compatible("highfreq_aligned")
True
>>> report.blocking("highfreq_aligned")
[]
>>> report.usable()
8
```

## What the axes mean

Six axes are checked, and it is worth knowing why each one is or is not fatal.

**`fs` — sampling rate.** Blocking for `highfreq_raw`. A 6 kHz window and a 44.1 kHz
window of the same duration are different lengths, so they cannot go in one tensor.
Not blocking for `highfreq_aligned`, because alignment resamples every cycle onto a
fixed grid: after it, both are `(n_cycles, cycle_size)`.

**`f0` — mains frequency.** Same story. 50 Hz and 60 Hz raw windows are different
lengths for the same cycle count; aligned, the period has been normalised away.

**`quantities` — what each channel stores.** *Partial*, not blocking. A waveform view
needs `v` and `i`; channels holding only `p` are skipped rather than fatal — unless
none are left. This is the common case in a real store, because a house's waveform
mains sits next to its 1/6 Hz meter channels, and a view selects between them.

**`appliance vocabulary`.** Never blocking — the merged label space is the union.
It is reported because *names that mean the same thing must match*. One corpus's
`microwave` and another's `oven` will become two columns unless you say otherwise.

**`dataset`.** Never blocking, but flagged, because a random split across two corpora
tests something different from what most people think it tests. See `CrossDataset` in
{doc}`splitting`.

**`supply voltage`.** Blocking for everything. This is the one alignment does not
fix and the one people forget.

## Merging

{func}`~nilmframe.store.merge_stores` writes a new store from several:

```{doctest}
>>> import tempfile, pathlib
>>> dst = pathlib.Path(tempfile.mkdtemp()) / "merged"
>>> merged = nf.merge_stores(
...     [store], dst,
...     rename={"microwave": "oven"},
...     normalize_voltage=230.0,
... )
>>> merged.appliances
['fridge', 'kettle', 'laptop', 'oven']
```

Three arguments carry the interesting behaviour.

`require` names axes that must agree before merging is permitted at all. It raises,
naming what differs, rather than producing a store that looks fine and is not:

```{code-block} python
merged = nf.merge_stores([uk, us], "stores/combined", require=["voltage", "f0"])
# ValueError: supply voltage differs across sources: [230.1, 119.8]
```

The useful entries are `voltage` and `f0`. Do not reach for `quantities`: it almost
always varies in a real store for the reason above.

`normalize_voltage` rescales every channel to a common RMS supply level. Current is
scaled inversely, so **active power is preserved** — the appliance still draws the same
watts, it just draws them at the reference voltage. That is the physically meaningful
normalisation, and it is what makes a 120 V corpus and a 230 V corpus comparable.

`rename` maps appliance aliases across every channel and every activation, so the two
corpora's vocabularies actually merge instead of doubling. For anything larger than a
one-off, use `taxonomy` instead — the next section.

The rules are recorded, so a result can be traced back to how its data was built:

```{doctest}
>>> merged.manifest["merge_rules"]["normalize_voltage"]
230.0
```

## Reconciling the labels

Voltage is the easy axis. The hard one is that every corpus names its loads its own
way, and a merge that does not reconcile them produces a store with twice the
vocabulary and half the data behind each label — a model trained on it learns that
`fridge` and `refrigerator` are different appliances.

REFIT alone makes the case. Its published label table has 48 distinct strings for
about twenty appliance types:

```{doctest}
>>> from nilmframe.readers.refit import APPLIANCES
>>> labels = sorted({name for house in APPLIANCES.values() for name in house.values()})
>>> len(labels)
48
>>> [name for name in labels if "ridge" in name or "irdge" in name]
['Firdge', 'Fridge', 'Fridge Freezer', 'Fridge(garage)', 'Fridge-Freezer', 'Fridge-Freezer(1)', 'Fridge-Freezer(2)']
```

Five different problems are mixed together there: a typo that shipped (`Firdge`,
and `Dishwaser` elsewhere in the table), separator variation, instance numbers,
circuits named for the appliance they feed (`Television Site`), and — genuinely —
a household member's initials (`PGM Computer`).

Case, separators and instance numbers are mechanical, and
{func}`~nilmframe.taxonomy.normalise` deals with them:

```{doctest}
>>> from nilmframe.taxonomy import normalise
>>> normalise("Fridge-Freezer(2)"), normalise("Fridge Freezer")
('fridge_freezer', 'fridge_freezer')
```

The rest is knowledge about the corpus, and knowledge has to be written down. That is
a {class}`~nilmframe.taxonomy.Taxonomy`: canonical name first, because that is the
direction you think in.

```{doctest}
>>> taxonomy = nf.Taxonomy({
...     "fridge": ["refrigerator", "Firdge"],
...     "washing_machine": ["Washing Machine", "washer"],
... })
>>> taxonomy.resolve("Refrigerator")
'fridge'
>>> taxonomy.resolve("Washing Machine(2)")
'washing_machine'
```

{func}`~nilmframe.taxonomy.default_taxonomy` ships a table covering the corpora this
package reads, which is a starting point rather than an authority:

```{doctest}
>>> shipped = nf.default_taxonomy()
>>> shipped.resolve("Dishwaser"), shipped.resolve("Television Site")
('dishwasher', 'television')
>>> shipped.category("dishwasher")
'wet'
```

### Look before you merge

This is the part that makes a taxonomy worth having over a dict. A merge writes a new
store and takes time; being able to see what it *would* do first is the difference
between a mapping you trust and one you hope about.
{meth}`~nilmframe.taxonomy.Taxonomy.report` is a dry run:

```{doctest}
>>> report = shipped.report(store)
>>> print(report[["label", "canonical", "category", "mapped"]].to_string(index=False))
    label canonical    category  mapped
   fridge    fridge        cold    True
   kettle    kettle     kitchen    True
   laptop    laptop electronics    True
microwave microwave     kitchen    True
```

Unmapped rows sort first, because those are the ones needing a decision.
{meth}`~nilmframe.taxonomy.Taxonomy.unmapped` is the short version:

```{doctest}
>>> nf.Taxonomy({"fridge": ["refrigerator"]}).unmapped(store)
['kettle', 'laptop', 'microwave']
```

The shipped table deliberately leaves some labels unmapped rather than guessing.
`K Mix` is a Kenwood kMix, which is a kettle in one product line and a stand mixer in
another; `Vivarium` and `Pond Pump` are what they say. Putting two different loads
under one label is the failure this is meant to prevent, so they surface here instead.

Extend it rather than replacing it:

```{doctest}
>>> mine = shipped.with_aliases(kettle=["K Mix"])
>>> mine.resolve("K Mix")
'kettle'
>>> shipped.resolve("K Mix") is None
True
```

### Applying it

Pass it to the merge. Labels it does not recognise are left alone rather than dropped
— an unrecognised label is recoverable, a discarded channel is not:

```{doctest}
>>> dst2 = pathlib.Path(tempfile.mkdtemp()) / "harmonised"
>>> harmonised = nf.merge_stores([store], dst2, taxonomy=shipped)
>>> harmonised.appliances
['fridge', 'kettle', 'laptop', 'microwave']
>>> harmonised.appliance_table["category"].tolist()
['cold', 'kitchen', 'electronics', 'kitchen']
```

That `category` column was `unknown` on every row before; a taxonomy is the only thing
that fills it.

Two more things it does that a flat dict cannot.

**It knows which corpus a label came from.** If one corpus's `washer` is a washing
machine and another's is a dishwasher, no global table can express that — `per_dataset`
overrides are consulted first:

```{doctest}
>>> ambiguous = nf.Taxonomy(
...     {"washing_machine": ["washer"]},
...     per_dataset={"odd_corpus": {"washer": "dishwasher"}},
... )
>>> ambiguous.resolve("washer", "refit"), ambiguous.resolve("washer", "odd_corpus")
('washing_machine', 'dishwasher')
```

**It refuses to resolve silently.** A `rename` with a typo in the key does nothing and
says nothing, which on a nine-corpus merge is the failure you find months later.
`strict=True` raises instead:

```{doctest}
>>> nf.default_taxonomy(strict=True).resolve("Vivarium")
Traceback (most recent call last):
    ...
KeyError: "no canonical name for 'Vivarium'. Add it to the taxonomy, or pass strict=False to leave unknown labels as they are."
```

`rename` still works and is applied *after* the taxonomy, so it stays the last word —
use it for the one-off that does not deserve a table. The resolved map is written into
the manifest, not the object, so a merged store still explains its own labels years
later when the table has moved on:

```{doctest}
>>> harmonised.manifest["merge_rules"]["taxonomy"]
{}
```

Empty here only because this store's labels were already canonical.

## Checking the result

Run compatibility again on what came out, and describe it:

```{doctest}
>>> nf.compatibility(merged).is_compatible("highfreq_aligned")
True
>>> merged.describe()
  appliance  channels  instances  brands     hours
0    fridge         2          2       2  0.001111
1    kettle         2          2       2  0.001111
2    laptop         1          1       1  0.000556
3      oven         1          1       1  0.000556
```

Channel ids are prefixed with their dataset by default, so two corpora that both call
a channel `house_1-mains` do not silently collide into one. Turn it off with
`prefix_with_dataset=False` only if you are certain the ids are already globally
unique.

## From the command line

```{code-block} bash
nilmframe compat stores/ukdale stores/plaid
nilmframe merge  stores/ukdale stores/plaid --dst stores/combined \
                 --require voltage --normalize-voltage 230
```

`compat` prints the same report before you commit to writing anything, which is the
order to do it in.
