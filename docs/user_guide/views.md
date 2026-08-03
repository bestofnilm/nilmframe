# Rate views

A *view* is the one place that decides how a stretch of stored samples becomes model
input. It answers three questions — which quantities to read, how wide a window is,
and what tensors come out — and nothing else in the pipeline answers them.

That concentration is the point. {class}`~nilmframe.data.WindowDataset` never touches
a raw signal itself; it asks the view. So swapping
{class}`~nilmframe.data.HighFreqView` for {class}`~nilmframe.data.LowFreqView`
changes the entire input pipeline without changing the dataset, the split, the loader,
the loss or the training loop. The rate a model operates at becomes a config flag.

The alternative — the usual arrangement, where a low-frequency experiment reads one
preprocessed file and a high-frequency experiment reads another — is what makes
"is 16 kHz worth it" so hard to answer honestly. The two arms end up differing by
preprocessing, by window boundaries and by which recordings survived, and the rate is
only one of the things that changed.

## HighFreqView

One window becomes `(n_cycles, cycle_size)` per quantity:

```{doctest}
>>> import torch, nilmframe as nf
>>> view = nf.HighFreqView(n_cycles=10, cycle_size=64, align="fitps")
>>> view
HighFreqView(n_cycles=10, cycle_size=64, align='fitps', f0=None, tol=0.2, slack=1.3)
```

Windows are specified in **cycles**, not samples. At 6 kHz on 50 Hz mains, ten cycles
is 1200 samples — but the view asks for more than that:

```{doctest}
>>> view.window_samples(6000.0, 50.0)
1716
```

That is `(10 + 1) × 120 × 1.3` — one spare cycle, times the `slack` factor. Mains
frequency drifts, and the window does not start on a zero crossing, so ten complete
cycles are not reliably found inside exactly ten cycles' worth of samples. The extra
guarantees they are. Whatever is left over after alignment is discarded, so the output
shape is fixed regardless:

```{doctest}
>>> store = nf.example_store()
>>> n = view.window_samples(6000.0, 50.0)
>>> raw = {q: torch.from_numpy(store.read_window("house_1-kettle", q, 0, n))
...        for q in ("v", "i")}
>>> out = view(raw, fs=6000.0, f0=50.0)
>>> {k: tuple(t.shape) for k, t in out.items() if t.ndim}
{'v': (10, 64), 'i': (10, 64), 'cycle_mask': (10,)}
>>> round(float(out["p_total"]), 1)
2924.8
```

Four keys come out. `v` and `i` are the aligned cycles. `cycle_mask` marks which of
the ten slots hold a real cycle rather than padding — when a drifting window yields
only nine, the tenth is padded and masked rather than duplicated, so the model never
learns from a cycle that was invented. And `p_total` is the window's active power,
**measured from this window's own voltage and current**. It is an input, not a label;
that distinction decides whether the reported metrics mean
anything.

### The unaligned control

Set `align=None` and you get the fixed-length control arm:

```{doctest}
>>> control = nf.HighFreqView(n_cycles=10, cycle_size=64, align=None)
>>> control.window_samples(6000.0, 50.0)
1716
```

Note that the number is *identical*. Both arms consume the same 1716 samples and
produce the same `(10, 64)` shape; only the resampling differs — aligned resamples
each detected cycle onto the grid, unaligned reshapes a fixed span. If the two arms
consumed different amounts of signal, the comparison would confound "alignment" with
"how much data each model saw", which is a real confound that has appeared in
published FITPS experiments. Keeping `window_samples` identical is deliberate.

### Options

`f0` overrides the channel's stored mains-frequency estimate. Leave it `None` unless
you know the estimate is wrong.

`tol` is the alignment tolerance as a fraction of the expected period — a candidate
cycle whose length is further than this from `fs/f0` is rejected as a false crossing.
Loosen it on noisy voltage, tighten it on clean.

`slack` is the oversizing factor above. Raise it if `cycle_mask` shows a lot of
padding, which means the window is not finding enough complete cycles.

## LowFreqView

The same window as a power series:

```{doctest}
>>> lf = nf.LowFreqView(rate_hz=1.0, n_steps=60)
>>> lf
LowFreqView(rate_hz=1.0, n_steps=60, quantity='active')
>>> out = lf(raw, fs=6000.0, f0=50.0)     # doctest: +SKIP
```

Sixty steps at 1 Hz is a sixty-second window, which at 6 kHz is 360000 samples:

```{doctest}
>>> lf.window_samples(6000.0, 50.0)
360000
```

The crucial property is that this series is **derived from the same waveform** the
high-frequency view reads, over the same window boundaries, by computing active power
per output step. It is not a separately preprocessed file. When both views are applied
to one store under one split, the only thing that differs between the two arms is the
rate — which is exactly the experiment people mean to run.

`LowFreqView` also works on channels that only ever stored power, which is most of
UK-DALE:

```{doctest}
>>> lf.supports({"p"}, fs=1.0, f0=50.0)
True
>>> nf.HighFreqView().supports({"p"}, fs=1.0, f0=50.0)
False
```

`WindowDataset` uses `supports` to *skip* channels a view cannot handle rather than
failing on them, so a mixed store yields whatever subset makes sense for the view you
chose.

## A fair rate comparison

Putting it together — one store, one split, two views:

```{code-block} python
store = nf.Store("stores/ukdale")
split = nf.LeaveHouseOut(test_size=0.3, seed=0).apply(store)

arms = {
    "lf":         nf.LowFreqView(rate_hz=1.0, n_steps=60),
    "hf-raw":     nf.HighFreqView(n_cycles=20, cycle_size=128, align=None),
    "hf-aligned": nf.HighFreqView(n_cycles=20, cycle_size=128, align="fitps"),
}
for name, view in arms.items():
    train = nf.WindowDataset(store, split.train, view=view)
    ...
```

Same store, same split object, same window boundaries. The three rows that come out
differ by the view and by nothing else. Holding everything else fixed turns this into a config
file so the shared parts cannot drift apart by accident.

## Writing your own

{class}`~nilmframe.data.View` is a structural protocol, not a base class. There is
nothing to subclass and nothing to register — implement four methods and you are a
view:

```{doctest}
>>> from nilmframe.data import View
>>> class Envelope:  # window of |i|, decimated: about the crudest useful view
...     def required_quantities(self, available):
...         return ('i',) if 'i' in available else ()
...     def supports(self, quantities, fs, f0):
...         return 'i' in quantities
...     def window_samples(self, fs, f0):
...         return int(fs)
...     def __call__(self, signals, fs, f0):
...         x = signals['i'].abs()
...         return {'p': x.reshape(-1, 100).mean(-1), 'p_total': x.mean()}
>>> isinstance(Envelope(), View)
True
```

One requirement is not optional: the returned dict must include `p_total`, the
measured active power of the window. The conservation term in
a conservation term is computed against it, and a view that omits it
silently disables the term that ties the model to a physical quantity.
