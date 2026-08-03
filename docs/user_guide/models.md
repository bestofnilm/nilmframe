# Reference models

Eleven architectures from ten papers, reimplemented so they can be compared.

The value here is not any one implementation — it is that they all take the same
input and return the same thing. Comparing two published models normally means
reading two repositories, reconciling two data formats and two output conventions,
and writing glue that is itself a source of error. Here it is a constructor call.

```{doctest}
>>> import torch, nilmframe.nn as nn_
>>> for name in ("seq2point", "unet", "bert4nilm"):
...     model = nn_.models.build(name, n_appliances=3, window=64)
...     print(f"{name:<10} {model.kind:<9} {tuple(model(torch.rand(1, 1, 64)).shape)}")
seq2point  seq2point (1, 3, 1)
unet       seq2seq   (1, 3, 64)
bert4nilm  seq2seq   (1, 3, 64)
```

## The contract

**In:** `(B, C, L)` — a batch of aggregate windows in watts. `C` is how many
measured quantities the model reads, almost always one. `L` is the window the
model declares.

**Out:** `(B, K, L_out)` — watts per appliance, clamped non-negative.

```{doctest}
>>> model = nn_.models.Seq2Seq(4, window=64)
>>> model.in_channels, model.window, model.n_appliances
(1, 64, 4)
>>> out = model(torch.rand(2, 1, 64) * 500)
>>> tuple(out.shape), bool((out >= 0).all())
((2, 4, 64), True)
```

A single-channel model also accepts `(B, L)` and adds the axis itself, because
writing `.unsqueeze(1)` at every call site is how that axis eventually ends up in
the wrong place.

Shapes are checked rather than broadcast. A window of the wrong length is a
mistake, and silently accepting it produces a number that looks fine:

```{doctest}
>>> model(torch.rand(2, 1, 32))
Traceback (most recent call last):
    ...
ValueError: Seq2Seq needs a window of 64, got 32
```

## Two kinds, and why the difference is visible

`L_out` is not always `L`. A model either reconstructs the whole window or
predicts its midpoint, and it says which:

```{doctest}
>>> nn_.models.Seq2Seq(3, window=64).kind, nn_.models.Seq2Seq(3, window=64).output_length
('seq2seq', 64)
>>> nn_.models.Seq2Point(3, window=65).kind, nn_.models.Seq2Point(3, window=65).output_length
('seq2point', 1)
```

The point framing is Zhang et al.'s and it exists for a concrete reason. A
sequence model must commit to a value at every step of the window, including the
edges where it has seen context on one side only; sliding the window and averaging
those commitments is what blurs a switching edge into a ramp. A point model makes
every prediction with the full window on both sides of it.

What it costs is arithmetic. A series of length `T` needs `T` forward passes
instead of `T / L`.

This asymmetry is deliberately not smoothed over. Padding a point model's output
back to `L` would be easy and would invent values the model never produced —
every metric computed over them would improve for no reason. So the shape stays
honest and the caller decides what to do with it.

## Building by name

```{doctest}
>>> sorted(nn_.models.MODELS)
['attention', 'bert4nilm', 'dae', 'electricity', 'seq2point', 'seq2seq', 'sgn', 'tpnilm', 'transfer', 'unet', 'wavenilm']
```

{func}`~nilmframe.nn.models.build` takes one of those names. The indirection is
what lets a sweep over architectures be a list of strings in a config file rather
than a chain of imports:

```{doctest}
>>> model = nn_.models.build("sgn", n_appliances=4, window=99, hidden=128)
>>> type(model).__name__, model.kind
('SGN', 'seq2point')
```

Everything past `n_appliances` is keyword-only in every model, so a misspelled
argument raises instead of silently taking a default. A misspelled *model* name
says what was available:

```{doctest}
>>> nn_.models.build("seq2pont", 4)
Traceback (most recent call last):
    ...
KeyError: "unknown model 'seq2pont'; available: attention, bert4nilm, dae,
electricity, seq2point, seq2seq, sgn, tpnilm, transfer, unet, wavenilm"
```

## The catalogue

Five share the seq2point stem — five convolutions of the widths in Zhang et al.'s
Table 1 — and differ in what sits on top. That grouping is not tidying: the papers
genuinely reuse each other's feature extractor.

| Name | Paper | Kind | What it is |
|---|---|---|---|
| `dae` | Kelly & Knottenbelt, BuildSys 2015 | seq2seq | The first deep NILM model. The aggregate *is* the appliance plus noise, so recovery is denoising |
| `seq2seq` | Zhang et al., AAAI 2018 | seq2seq | Reconstructs the window |
| `seq2point` | Zhang et al., AAAI 2018 | seq2point | Predicts the midpoint. The field's default baseline |
| `sgn` | Shin et al., AAAI 2019 | seq2point | Regression gated by an on/off classifier |
| `transfer` | D'Incecco et al., IEEE TSG 2020 | seq2point | seq2point plus a transfer protocol |
| `unet` | Faustine et al., BuildSys 2020 | seq2seq | 1D U-Net; skips put back the edge that pooling destroys |
| `wavenilm` | Harell et al., ICASSP 2019 | seq2seq | Causal dilated convolutions over complex power |
| `tpnilm` | Massidda et al., Appl. Sci. 2020 | seq2seq | Temporal pooling pyramid over several scales |
| `attention` | Piccialli & Sudoso, Energies 2021 | seq2seq | Convolutions for local shape, attention for the rest |
| `bert4nilm` | Yue et al., BuildSys 2020 | seq2seq | Bidirectional transformer, masked pretraining |
| `electricity` | Sykiotis et al., Sensors 2022 | seq2seq | Transformer with a replaced-token objective |

None of them is trained. Constructing one gives random weights, and the library
ships no checkpoints.

## Where they diverge from the common case

**SGN returns its gate.** The gating is the paper's contribution: a regression
head trained on windows where the appliance is off spends its capacity learning
to output zero. A gate that has collapsed to always-on is the failure mode, and it
is invisible in the power alone — so it is reachable:

```{doctest}
>>> sgn = nn_.models.SGN(3, window=99, hidden=64)
>>> gate = sgn.gate(torch.rand(2, 1, 99) * 500)
>>> tuple(gate.shape), bool(((gate >= 0) & (gate <= 1)).all())
((2, 3, 1), True)
```

**WaveNILM reads complex power.** It is the one model here built for `(P, Q)`
rather than watts alone, which is what separates loads drawing the same active
power at different power factors:

```{doctest}
>>> wave = nn_.models.WaveNILM(3, window=64, in_channels=2, width=16, layers=4)
>>> tuple(wave(torch.rand(2, 2, 64) * 500).shape)
(2, 3, 64)
```

Its convolutions are causal — padded on the left only — because the paper's case
is online disaggregation, where a model that reads the future cannot be deployed.

**The two transformers carry their pretraining objective as a method**, so it is
runnable rather than described. BERT4NILM hides steps and asks for them back:

```{doctest}
>>> bert = nn_.models.BERT4NILM(3, window=64, width=32, heads=2, layers=1)
>>> corrupted, hidden = bert.mask(torch.rand(2, 1, 64))
>>> tuple(corrupted.shape), tuple(hidden.shape)
((2, 1, 64), (2, 64))
```

ELECTRIcity replaces steps with a small generator's guesses and asks which were
replaced. The generator is deliberately weak: a strong one produces replacements
the discriminator cannot detect, and the task stops teaching anything.

```{doctest}
>>> electra = nn_.models.ELECTRIcity(3, window=64, width=32, heads=2, layers=1)
>>> corrupted, replaced = electra.corrupt(torch.rand(2, 1, 64))
>>> tuple(replaced.shape)
(2, 64)
```

**TransferNILM freezes.** Architecturally it *is* seq2point — the contribution is
the protocol, and the protocol is what you reproduce:

```{doctest}
>>> transfer = nn_.models.TransferNILM(3, window=99, hidden=64)
>>> transfer.freeze_features()
>>> any(p.requires_grad for p in transfer.features.parameters())
False
>>> all(p.requires_grad for p in transfer.head.parameters())
True
```

There is no method for fine-tuning at a lower rate, because that is an optimiser
argument rather than a property of the model.

## Scaling the input

Every one of these papers standardises its input, and the constants differ per
corpus. A model that silently assumed the authors' UK-DALE statistics will look
broken on REFIT, for a reason that takes a day to find. So it is an explicit
argument:

```{doctest}
>>> scale = nn_.models.Standardiser(mean=500.0, std=800.0)
>>> model = nn_.models.Seq2Point(3, window=65, standardiser=scale, hidden=64)
>>> tuple(model(torch.full((1, 1, 65), 500.0)).shape)
(1, 3, 1)
```

The constants are buffers, not Python floats, so they travel with a checkpoint.
Passing `None` feeds the model raw watts, which is right only when you have
scaled them already.

## From a dataset

{meth}`~nilmframe.nn.models.NILMModel.predict` takes a batch and reads signals
only — `p_total` if present and `p` otherwise, never `power` or `presence`, which
are labels:

```{doctest}
>>> model = nn_.models.Seq2Seq(3, window=64, hidden=64)
>>> batch = {"p": torch.rand(2, 64) * 400, "power": torch.zeros(2, 3, 64)}
>>> out = model.predict(batch)
>>> sorted(out), tuple(out["power"].shape), tuple(out["total"].shape)
(['power', 'total'], (2, 3, 64), (2, 64))
```

A batch with no aggregate at all raises rather than reaching for a label:

```{doctest}
>>> model.predict({"power": torch.zeros(2, 3, 64)})
Traceback (most recent call last):
    ...
KeyError: "batch carries no aggregate: expected 'p_total' or 'p'"
```

## What is not here

**Training.** No loop, no losses, no checkpoints. These are reference
architectures; the objective you train them under is yours.

**Fidelity claims.** Each is a reimplementation from its paper with the paper's
stated defaults, not a port of the authors' code. Where a paper leaves a choice
unstated the choice is marked in the class that makes it. Expect the same shape of
result, not the same digits.
