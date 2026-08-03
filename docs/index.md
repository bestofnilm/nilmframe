---
sd_hide_title: true
---

# nilmframe

:::{div} nf-hero
```{image} _static/logo.svg
:alt: nilmframe
:class: nf-hero-mark
```

<h1 class="nf-hero-name">nilmframe</h1>

PyTorch-native end-to-end non-intrusive load monitoring.
:::

```{code-block} python
import torch
import nilmframe as nf

store = nf.example_store()                       # or nf.Store("~/.nilm/ukdale")
view  = nf.HighFreqView(n_cycles=20, cycle_size=128, align="fitps")
split = nf.LeaveHouseOut(test_size=0.3, seed=0).apply(store)

train  = nf.WindowDataset(store, split.train, view=view)
loader = torch.utils.data.DataLoader(train, batch_size=64, collate_fn=nf.collate_windows)
```

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Getting started
Install it, look at a measurement, build a dataset, feed a loader. Fifteen
minutes, no download.
+++
{doc}`getting_started`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` User guide
The whole path the data takes, chapter by chapter, with the reasoning.
Fifteen chapters, every example runnable.
+++
{doc}`user_guide/index`
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API reference
Every public class and function, with signature, arguments and an example.
+++
{doc}`api/nilmframe`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Design notes
Why the store is lazy, why the aggregate is an input, why a rate is a view
rather than a pipeline.
+++
{doc}`concepts`
:::
::::

## The guide at a glance

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Data
{doc}`Datasets <user_guide/datasets_tour>` ·
{doc}`Downloading <user_guide/downloading>` ·
{doc}`Loading data <user_guide/data_loading>` ·
{doc}`Measurements <user_guide/measurements>` ·
{doc}`Combining datasets <user_guide/combining>` ·
{doc}`Rate views <user_guide/views>` ·
{doc}`Datasets and loaders <user_guide/datasets>` ·
{doc}`Splitting <user_guide/splitting>` ·
{doc}`Augmentation <user_guide/augmentation>`
:::

:::{grid-item-card} Signal
{doc}`Cycle alignment <user_guide/alignment>` ·
{doc}`Representations <user_guide/representations>` ·
{doc}`Event detection <user_guide/event_detection>`
:::

:::{grid-item-card} Models
{doc}`Reference models <user_guide/models>` ·
{doc}`Evaluation <user_guide/evaluation>` ·
{doc}`Command line <user_guide/cli>`
:::
::::

## Install

```bash
pip install -e ".[all]"     # or: pip install -e .   for the light core
```

The core needs only `torch`, `numpy`, `pandas` and `pyarrow`. There is **no compiled
extension**: cycle alignment is pure PyTorch, so it is batched, runs on the GPU as part
of the model, and installs everywhere.

## Three ideas hold the design together

**One canonical store.** Tabular metadata, memory-mapped signals. Nothing is
materialised until a window is asked for, so a corpus larger than memory is ordinary.

**Many rate views.** The low-frequency series a 1 Hz model sees is *derived from* the
same waveform a 16 kHz model sees, over the same window boundaries. Comparing them is a
config flag rather than two preprocessing pipelines.

**The aggregate is an input.** It is measured from the signal, never summed from the
labels — so a number reported here is one the model could produce at inference, on a
meter that has no labels at all.

## Every example here runs

Docstring examples execute against a small built-in corpus,
{func}`nilmframe.example_store`, which is generated on first use and downloads nothing.
`make doctest` and `pytest --doctest-modules` both run them, so an example that stops
working fails the build instead of quietly misleading you.

```{toctree}
:hidden:
:caption: Getting started

getting_started
concepts
```

```{toctree}
:hidden:
:caption: User guide
:numbered:

user_guide/downloading
user_guide/datasets_tour
user_guide/data_loading
user_guide/measurements
user_guide/combining
user_guide/views
user_guide/datasets
user_guide/splitting
user_guide/augmentation
user_guide/alignment
user_guide/representations
user_guide/event_detection
user_guide/models
user_guide/evaluation
user_guide/cli
```

```{toctree}
:hidden:
:caption: API reference

api/nilmframe
api/nilmframe.store
api/nilmframe.data
api/nilmframe.nn
api/nilmframe.nn.models
api/nilmframe.eval
api/nilmframe.readers
api/nilmframe.sources
api/nilmframe.measurement
api/nilmframe.compat
api/nilmframe.taxonomy
api/nilmframe.docdata
```
