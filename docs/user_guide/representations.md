# Signal representations

Almost every NILM paper's contribution is really a claim about *representation*: that
some transformation of the raw waveform exposes the information a classifier needs. V-I
trajectory images, harmonic vectors, spectrograms, distance matrices — these are the
things people compare.

So they are all here as `nn.Module` transforms, with the same interface, so comparing
them is a line of code rather than a reimplementation. They are ordinary modules: they
compose, they run on the GPU, they are differentiable, and they can sit inside the
model rather than in a preprocessing script.

Everything on this page is imported from `nilmframe.nn`.

```{doctest}
>>> import torch, nilmframe as nf
>>> import nilmframe.nn as nn_
>>> m = nf.example_measurement().aligned(cycle_size=128)
>>> tuple(m.v.shape), tuple(m.i.shape)
((23, 128), (23, 128))
```

## Time domain

The cheap ones. Length reduction, mostly, and the choice among them is about what you
are willing to lose.

**{class}`~nilmframe.nn.PAA`** — piecewise aggregate approximation. Average within
equal-width bins:

```{doctest}
>>> tuple(nn_.PAA(32)(torch.randn(2, 256)).shape)
(2, 32)
```

Averaging is a lowpass, so PAA keeps the envelope and discards transients. Right when
the signature is in the shape, wrong when it is in the switching edge.

**{class}`~nilmframe.nn.Downsample`** — strided selection instead of averaging. Keeps
transients, aliases everything above the new Nyquist. The complementary trade.

```{doctest}
>>> tuple(nn_.Downsample(64)(torch.randn(2, 256)).shape)
(2, 64)
```

**{class}`~nilmframe.nn.Patchify`** — no reduction at all, a reshape into overlapping
or non-overlapping patches, for transformer-style models that want tokens:

```{doctest}
>>> tuple(nn_.Patchify(32)(torch.randn(2, 256)).shape)
(2, 8, 32)
```

**{class}`~nilmframe.nn.StandardScale`** — normalisation with fixed statistics you
supply, so training and inference share them rather than each computing its own from
whatever batch happened to be present.

## The V-I trajectory

Plot instantaneous current against instantaneous voltage over one cycle and you get a
closed orbit. A resistor gives a straight line; an inductive load gives an ellipse; a
switched-mode power supply gives a narrow spike-and-return. The shape is characteristic
enough that rasterising it into an image and running a CNN over it is a whole family of
published methods.

Here is the claim on real data — three PLAID appliances, one cycle each, current
normalised by its own peak so the shapes rather than the scales are compared:

```{image} ../_static/datasets/vi-signatures-light.png
:class: only-light
:alt: V-I trajectories of a compact fluorescent lamp, a fridge and a hairdryer.
```

```{image} ../_static/datasets/vi-signatures-dark.png
:class: only-dark
:alt: V-I trajectories of a compact fluorescent lamp, a fridge and a hairdryer.
```

The hairdryer is a heating element, so current follows voltage and the orbit collapses
towards a line. The fridge's motor is inductive: current lags, and the line opens into
an S-shaped loop. The lamp's supply draws nothing until the rectifier conducts near the
voltage peak, which is the flat shelf and the near-vertical jump. Three appliances, three
shapes, before any model has been trained.

Normalising is what makes them comparable — the hairdryer's peak is 18.2 A against the
lamp's 1.12 A, so on shared axes the lamp would be a dot.
{class}`~nilmframe.nn.VITrajectory` rasterises the un-normalised orbit and keeps the
magnitude in its third channel.

```{doctest}
>>> tuple(nn_.VITrajectory(image_size=32)(m.v, m.i).shape)
(23, 3, 32, 32)
```

Three channels, not one: occupancy (did the orbit pass through this cell), local
trajectory slope, and instantaneous power. The extra two carry direction and magnitude
information that a pure occupancy mask throws away.

One cycle in, one image out — twenty-three cycles gives twenty-three images. This is a
per-cycle representation, which is exactly why it wants aligned input; rasterising an
unaligned cycle smears the orbit around the phase offset.

The rasterisation is a scatter, done out-of-place, so gradients flow through it. That
matters if you want to learn anything upstream of the image.

## The distance matrix

Self-similarity: entry `(m, n)` is `|x_m - x_n|`.

```{doctest}
>>> tuple(nn_.DistanceMatrix()(torch.randn(2, 16)).shape)
(2, 16, 16)
```

Periodic structure appears as diagonal banding, which is what makes it a useful 2-D
input to a convolutional encoder. The cost is quadratic in input length, so put a
{class}`~nilmframe.nn.PAA` in front of it if `T` is more than a few hundred.

## Frequency domain

**{class}`~nilmframe.nn.HarmonicLowpass`** — keep the first `n` harmonics, drop the
rest, return to the time domain:

```{doctest}
>>> tuple(nn_.HarmonicLowpass(8)(m.i).shape)
(23, 128)
```

Denoising that respects the physics: harmonics are what an appliance actually produces,
so keeping sixteen of them and discarding the rest removes measurement noise without
removing signature.

**{class}`~nilmframe.nn.ReIm`** — the harmonic coefficients themselves, as
interleaved real and imaginary parts:

```{doctest}
>>> tuple(nn_.ReIm(8)(m.i).shape)
(23, 16)
```

Eight harmonics become sixteen numbers. Real and imaginary rather than magnitude and
phase because phase wraps, and a network learning across a wrap discontinuity is a
network learning something unnecessary.

**{class}`~nilmframe.nn.Spectrogram`** — short-time Fourier magnitude, the standard
time-frequency view:

```{doctest}
>>> tuple(nn_.Spectrogram(window_size=64, hop_size=16)(torch.randn(2, 256)).shape)
(2, 33, 17)
```

Unlike the harmonic transforms this does not need aligned input, which makes it the
natural choice for a raw or a low-frequency arm.

**{class}`~nilmframe.nn.DFIA`** — double Fourier integral analysis. Build the outer
product `v_m · i_n` — the instantaneous power matrix — and take its 2-D FFT:

```{doctest}
>>> tuple(nn_.DFIA(n_fft=(4, 4))(torch.randn(2, 32), torch.randn(2, 32)).shape)
(2, 2, 4, 4)
```

A joint voltage-current spectral signature. Expensive, and the `n_fft` crop is how you
keep it affordable — the informative content is at low order.

## Fryze decomposition

{class}`~nilmframe.nn.Fryze` splits current into the part collinear with the voltage
and the remainder:

```{doctest}
>>> tuple(nn_.Fryze()(m.v, m.i).shape)
(23, 3, 128)
```

The active part is `i_a = P / V_rms² · v` — the current that actually does work. The
remainder `i - i_a` carries the reactive and distortion content. Three channels come
out: the two parts plus the voltage.

The reason this is a good default input is that it separates *how much power* from
*what shape*, and most of what distinguishes appliance classes is shape. A 2 kW kettle
and a 2 kW heater have nearly identical active parts and different non-active parts.
{class}`~nilmframe.nn.Fryze` is the transform that does it.

## Composing

They are modules, so `nn.Sequential` works:

```{doctest}
>>> import torch.nn as tnn
>>> pipe = tnn.Sequential(nn_.PAA(64), nn_.DistanceMatrix())
>>> tuple(pipe(torch.randn(4, 512)).shape)
(4, 64, 64)
```

Reduce first, then build the quadratic thing. The other order allocates a 512×512
matrix per item and then throws most of it away.

## Choosing

There is no universal answer, which is the honest version, and there is a shape to the
trade-off worth knowing.

If the signature is in **cycle shape** — motors, switched-mode supplies, anything with
distinctive distortion — use an aligned representation: V-I trajectory, Fryze, or
harmonics. These need `align="fitps"` upstream to mean anything.

If it is in **timing** — duty cycles, ramp rates, a fridge compressor's periodicity —
use a spectrogram or a distance matrix over a longer window, and alignment stops
mattering.

If you do not know, the useful thing about this library is that finding out is an
ablation, not a rewrite. Swap the adapter, keep everything else, and read the two rows
off the results table.
