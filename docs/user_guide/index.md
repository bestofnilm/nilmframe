---
orphan: true
---

# User guide

This guide is meant to be read. It walks the whole path a disaggregation
experiment takes — from a folder of vendor files to a number you can defend —
and explains the reasoning at each step, not only the call.

Every example runs. They execute against {func}`nilmframe.example_store`, a small
two-house corpus generated on first use that downloads nothing, and `make doctest`
executes each one on every build. An example that stops working fails the build.

If you are looking for the signature of one particular thing, the
{doc}`API reference <../api/nilmframe>` is the place. If you are looking for how the
pieces fit together, start here.

## The chapters, in order

1. {doc}`downloading` — a slice of a public corpus without the whole corpus
2. {doc}`datasets_tour` — what each corpus holds, at what rate
3. {doc}`data_loading` — getting your own data in
4. {doc}`measurements` — the object everything else returns
5. {doc}`combining` — merging corpora, and reconciling their labels
6. {doc}`views` — one store, several sampling rates
7. {doc}`datasets` — windows, targets and loaders
8. {doc}`splitting` — leakage, and how to avoid it
9. {doc}`augmentation` — what is safe to synthesise
10. {doc}`alignment` — cycles rather than samples
11. {doc}`representations` — what to feed a model
12. {doc}`event_detection` — deciding when something switched
13. {doc}`models` — fourteen reference architectures under one call signature
14. {doc}`evaluation` — what the metrics actually measure
15. {doc}`cli` — the same things without Python

## Where to start

You do not have to read it in order.

**Choosing a corpus.** {doc}`datasets_tour` shows what each of the nine datasets
contains, at what resolution, and what it cannot be used for.

**Just want to see data.** {doc}`measurements` is the shortest path from nothing to
a plot. {doc}`downloading` fetches a slice of a public corpus without the whole
corpus, and {doc}`data_loading` covers getting your own data in.

**Comparing sampling rates.** {doc}`views` is the chapter that matters, and
{doc}`alignment` explains why a high-rate window is defined in cycles rather than
samples.

**Worried about the number you are about to publish.** {doc}`splitting` on leakage,
{doc}`evaluation` on what the metrics actually measure.

**Feeding a model.** {doc}`representations` for what to put in front of one,
{doc}`models` for the architectures themselves.

## Notation

Through the guide, `B` is a batch, `K` the number of appliance classes, `T` a number
of time steps, `C` a number of cycles and `S` the samples per cycle. Signals are
`v` (volts), `i` (amperes) and `p` (watts). Shapes are written as tuples, so
`(B, C, S)` is a batch of `C` cycles of `S` samples each.
