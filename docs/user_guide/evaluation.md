# Evaluation

Disaggregation has no single metric, because it is not a single task. Detection and
regression fail differently, and a model can be excellent at one while being useless at
the other. Reporting one number hides that; reporting eleven without saying which
matter is not much better.

{class}`~nilmframe.eval.Evaluator` runs a standard collection at once and returns a
flat dict, and `HEADLINE_COLUMNS` marks which of them are higher-is-better so a results
table can be read without a lookup.

```{doctest}
>>> import torch, nilmframe as nf
>>> from nilmframe.eval import Evaluator, HEADLINE_COLUMNS
>>> ev = Evaluator(4, ["fridge", "kettle", "laptop", "microwave"])
>>> sorted(HEADLINE_COLUMNS)
['f1_macro', 'mae', 'mcc', 'modified_f1', 'modified_jaccard', 'nde', 'sae', 'teca']
```

## Running it

`update` takes a prediction dict and the batch it came from, so it plugs straight into
an evaluation loop:

```{doctest}
>>> pred = {"presence": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool),
...         "power": torch.tensor([[100., 0., 50., 0.], [0., 2000., 0., 0.]]),
...         "probability": torch.tensor([[.9, .1, .8, .2], [.2, .95, .1, .05]]),
...         "total": torch.tensor([150., 2000.])}
>>> batch = {"presence": torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 0.]]),
...          "power": torch.tensor([[110., 0., 45., 0.], [0., 1900., 0., 0.]]),
...          "p_total": torch.tensor([155., 1900.])}
>>> ev.update(pred, batch)
>>> results = ev.compute()
>>> round(results["f1_macro"], 3), round(results["nde"], 3)
(1.0, 0.056)
```

In a loop:

```{code-block} python
ev = nf.Evaluator(store.n_appliances, store.appliances)
model.eval()
for batch in val_loader:
    ev.update(model.predict(batch), batch)
results = ev.compute()
```

Note `model.predict(batch)`, not `model(batch)`. `predict` applies the thresholds and
the optional measured-total rescaling, and it runs under `no_grad` with the module
temporarily in eval mode.

## Detection metrics

**`f1_macro`** averages per-appliance F1 with equal weight per class. This is the
number to report when rare appliances matter as much as common ones, which is usually.

**`f1_micro`** pools all decisions first, so it is dominated by whichever appliance
appears most. Report both, and a large gap between them tells you the model is carrying
its score on the common classes.

**`f1_per_appliance:<name>`** breaks it out. Read this one:

```{doctest}
>>> {k: v for k, v in results.items() if k.startswith("f1_per_appliance")}
{'f1_per_appliance:fridge': 1.0, 'f1_per_appliance:kettle': 1.0, 'f1_per_appliance:laptop': 1.0, 'f1_per_appliance:microwave': 0.0}
```

Microwave scores 0.0 — it never appears in this two-window batch, so its F1 is
undefined and reported as zero rather than silently dropped. That is exactly the case
{meth}`~nilmframe.data.Split.summary` warns about in {doc}`splitting`: a class absent
from the evaluation fold drags `f1_macro` down for reasons that have nothing to do with
the model.

**`mcc`** — Matthews correlation coefficient. Unlike F1 it accounts for true negatives,
which makes it the more honest headline on imbalanced data, where most appliances are
off in most windows.

## Regression metrics

**`mae`** — mean absolute error in watts. Interpretable, and dominated by the largest
appliances.

**`nde`** — normalised disaggregation error, total absolute error over total true
energy. Scale-free, so a fridge and an oven are comparable; a 50 W error on each is not
the same mistake, and NDE is the metric that knows it.

**`sae`** — signal aggregate error, the relative error of *total energy per appliance*
over the whole evaluation. A model can have a poor MAE and an excellent SAE by getting
the energy right while getting the timing wrong, which for a billing application is
fine and for a control application is not.

**`teca`** — total energy correctly assigned, `1 − Σ|y − ŷ| / (2 Σ y)`. The most
commonly reported NILM headline. Windows with no load at all are skipped rather than
producing a division by zero.

## Joint metrics

The interesting ones, because they refuse to let detection and regression be scored
separately.

**`modified_f1`** counts a detection as correct only if the power estimate is within
`delta` of the truth. Detecting the kettle and predicting 200 W for a 2000 W load is
not a success, and plain F1 says it is.

**`modified_jaccard`** is the same idea over the intersection-over-union of the on
periods, with an absolute watt tolerance.

```{doctest}
>>> round(results["modified_f1"], 3), round(results["modified_jaccard"], 3)
(1.0, 0.667)
```

Notice these two disagree on the same predictions. That is the point of having both.

## Calibration

**`calibration`** is expected calibration error over the presence probabilities. A model
whose 0.9 predictions are right 90 % of the time is calibrated; one whose 0.9
predictions are right 60 % of the time is not, and any downstream decision that uses a
threshold is built on sand.

Worth reporting, rarely reported.

## Individual metrics

Each is a `torchmetrics.Metric`, usable on its own:

```{doctest}
>>> from nilmframe.eval import TECA, NormalisedDisaggregationError
>>> metric = TECA()
>>> metric.update(torch.tensor([[100., 50.]]), torch.tensor([[100., 50.]]))
>>> round(float(metric.compute()), 3)
1.0
```

The `update`/`compute` split means they accumulate across batches correctly rather than
averaging per-batch averages, which is wrong for anything with a ratio in it.

## Reporting

{func}`~nilmframe.eval.format_table` prints a results frame with the direction arrows
attached:

```{doctest}
>>> import pandas as pd
>>> from nilmframe.eval import format_table, compare
>>> table = pd.DataFrame([{"name": "lf", "f1_macro": 0.71, "nde": 0.42},
...                       {"name": "hf", "f1_macro": 0.83, "nde": 0.29}])
>>> print(format_table(table, precision=3))
name  f1_macro ^  nde v
  lf       0.710  0.420
  hf       0.830  0.290
```

The `^` and `v` come from `HEADLINE_COLUMNS`. `f1_macro` higher is better; `nde` lower
is better. Nobody has to remember which.

{func}`~nilmframe.eval.compare` diffs arms against a named baseline:

```{doctest}
>>> compare(table, baseline="lf")
  name  f1_macro   nde
0   lf      0.00 -0.00
1   hf      0.12  0.13
```

The numbers are **improvements**, already signed so that positive is better on every
column. `hf` gained 0.12 F1 and 0.13 NDE — the NDE figure is positive even though the
raw value fell, because for NDE falling is winning. Nobody has to keep the direction of
eleven metrics in their head while reading a table.

## Open set

When the split held classes out (see {doc}`splitting`), pass `open_set=True` and the
collection adds {class}`~nilmframe.eval.UnknownAUROC` — how well the model separates
windows containing an unknown appliance from those that do not, without any threshold
being chosen.

The per-class metrics still only cover the known vocabulary, which is correct: there is
no per-class score to give for a class the model was never told about.

## What to report

Two detection numbers (`f1_macro` and `mcc`), two regression numbers (`nde` and
`teca`), one joint number (`modified_f1`), and the per-appliance breakdown. Plus the
split manifest, so a reader knows what was actually held out — which changes these
numbers more than any architectural choice does.
