# Design notes

Five decisions the rest of the library follows from. Each is expanded in the
{doc}`user guide <user_guide/index>`; this page is the short version, for deciding
whether the design suits what you are doing.

## The store is lazy

Metadata is tabular; signals are `.npy` opened with `mmap_mode="r"`. Reading a window
touches only the pages it needs, so the cost of a window into a house-year is the cost
of the window.

Everything a dataset object would normally carry as a method — `submetered`,
`drop_rare`, `filter`, `groupby` — is a dataframe expression on
{attr}`~nilmframe.store.Store.channels` instead:

```{doctest}
>>> import nilmframe as nf
>>> store = nf.example_store()
>>> store.channels.query("kind == 'submeter' and brand == 'acme'").appliance.tolist()
['kettle', 'fridge', 'laptop']
```

There is no API to learn for that, and no method missing when you want something the
author did not anticipate.

→ {doc}`user_guide/data_loading`

## Windows are defined in cycles, not samples

Mains frequency drifts, so a fixed-length slice of a waveform is not phase-aligned with
the next. {func}`~nilmframe.nn.cycle_align` locates rising zero crossings of the voltage
to sub-sample precision and resamples each cycle onto a fixed grid.

Specifying a window in *cycles* makes alignment produce a fixed output shape, which is
what removes ragged batching from the whole pipeline.

It also has a consequence for combining corpora: after alignment, a 6 kHz 60 Hz
recording and a 44.1 kHz 50 Hz one are the same shape and the same phase. Sampling rate
stops being a blocking difference — which is why {func}`nilmframe.compatibility`
reports per intended use rather than as a verdict.

→ {doc}`user_guide/alignment`, {doc}`user_guide/combining`

## One waveform, many rate views

The low-frequency series a 1 Hz model sees is *derived from* the same stored waveform a
16 kHz model sees, over the same window boundaries. Not a separate preprocessed file.

That is what makes "is high frequency worth it" answerable. In the usual arrangement the
two arms differ by preprocessing, by window boundaries and by which recordings survived,
and the rate is only one of the things that changed. Here the rate is a config flag and
everything else is held fixed by construction.

→ {doc}`user_guide/views`

## The aggregate is an input, not a label

`p_total` is computed from the window's own voltage and current:

```{doctest}
>>> sorted(nf.example_measurement().aligned(cycle_size=64).batch())
['cycle_mask', 'i', 'p_total', 'v']
```

Those keys are all a model may read. `presence`, `power` and `power_mask` are labels,
and a batch built from a measurement does not carry them at all.

That distinction is the whole reason the metrics can be trusted. A model rescaled by the
*sum of its labels* cannot be deployed, and its scores are inflated by an amount nobody
can estimate after the fact. Using the measured aggregate is legitimate — a meter
reports it. Using the labels is not.

## Open set is structural

Held-out appliance classes get **no column** in the label space, so a model has nowhere
to put them and must learn to say "not mine". Windows containing one are flagged
`is_unknown`, contribute no presence target and no attributable power, and are scored by
{class}`~nilmframe.eval.UnknownAUROC`.

```{doctest}
>>> split = nf.UnseenAppliance(unknown=["microwave"], seed=0).apply(store)
>>> split.manifest["known"], split.manifest["unknown"]
(['fridge', 'kettle', 'laptop'], ['microwave'])
```

The usual alternative — an "other" class trained on whatever was left over — teaches the
model that unknown has a signature, which it does not.

→ {doc}`user_guide/splitting`

## Documentation that cannot rot

Every example in this documentation is executed on every build, against the same
generated corpus. `make doctest` runs the narrative pages and the docstrings together;
`pytest --doctest-modules` runs the docstrings again in CI.

An example that stops being true fails the build. That is the only mechanism that
actually keeps documentation honest over a codebase's life, and it is why the examples
here are small and concrete rather than illustrative pseudocode.
