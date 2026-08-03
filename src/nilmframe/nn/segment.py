"""Event detection: where does the load change.

Stage one of an event-based pipeline. Each detector here decides *whether and
when* something switched, and nothing more -- no appliance identity, which is
what the models in :mod:`nilmframe.nn` and the metrics in :mod:`nilmframe.eval`
are for.

Every detector is an :class:`~torch.nn.Module` that takes a power envelope,
``(T,)`` or ``(B, T)``, and returns a boolean mask of the same shape. A mask
rather than indices, because a list of indices has a length that depends on the
*contents* of the input, so a batch of them goes ragged and cannot be stacked.
That one convention is what makes the detectors interchangeable, and it is why
adding another one requires no change anywhere else.

Pure torch and batched, like the rest of :mod:`nilmframe.nn`. The user guide's
event-detection page has the catalogue -- what each asks, its parameters, and how
they score against each other on submetered data.

Replaces ``legacy/events/``, which was five empty files -- ``bayes.h``,
``pereira.h``, ``volker.h``, ``z_score.h``, ``bindings.cpp`` -- and one empty
``__init__.py``. Nothing was ever written, but the dataset layer already called
into it: ``HighFreqDataset.split_by_events(detector)`` expected an object with a
``find_changepoints`` method that did not exist anywhere.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "ActiveSectionDetector",
    "AdaptiveThresholdDetector",
    "CusumDetector",
    "GLRDetector",
    "GoodnessOfFitDetector",
    "MultivariateDetector",
    "ZScoreDetector",
    "segments_from_mask",
]


def _as_batched(x: Any) -> tuple[Tensor, bool]:
    """Coerce to a tensor and add a batch axis if there is not one.

    Coerces rather than demanding a tensor because a detector is the first thing
    a new user calls, often on a numpy array they read themselves, and
    ``'numpy.ndarray' object has no attribute 'unsqueeze'`` is a poor way to
    learn that this library is built on torch. :class:`~nilmframe.Measurement`
    already accepts the same inputs; ``torch.as_tensor`` does not copy a numpy
    array, so the convenience is free.
    """
    if not isinstance(x, Tensor):
        x = torch.as_tensor(x)
        if not x.is_floating_point() and x.dtype is not torch.bool:
            x = x.float()
    return (x.unsqueeze(0), True) if x.ndim == 1 else (x, False)


def _running_stats(x: Tensor, window: int) -> tuple[Tensor, Tensor]:
    """Causal running mean and standard deviation over the last ``window`` samples."""
    padded = torch.nn.functional.pad(x.unsqueeze(1), (window - 1, 0), mode="replicate")
    mean = torch.nn.functional.avg_pool1d(padded, window, stride=1).squeeze(1)
    mean_square = torch.nn.functional.avg_pool1d(padded.pow(2), window, stride=1).squeeze(1)
    return mean, (mean_square - mean.pow(2)).clamp_min(0).sqrt()


def segments_from_mask(mask: Tensor, min_length: int = 1) -> list[list[tuple[int, int]]]:
    """Turn a per-sample event mask into ``[start, stop)`` spans, per batch item.

    Args:
        mask: ``(B, T)`` or ``(T,)`` boolean events.
        min_length: drop segments shorter than this.

    Returns:
        One list of spans per item.

    Example:
        >>> mask = torch.zeros(10, dtype=torch.bool)
        >>> mask[3] = mask[7] = True
        >>> nf.nn.segments_from_mask(mask)
        [(0, 3), (3, 7), (7, 10)]
        >>> nf.nn.segments_from_mask(mask, min_length=4)
        [(3, 7)]
    """
    mask, squeezed = _as_batched(mask)
    out: list[list[tuple[int, int]]] = []
    length = mask.shape[-1]
    for row in mask:
        edges = [0, *(int(k) for k in row.nonzero().flatten()), length]
        spans = [(a, b) for a, b in pairwise(edges) if b - a >= min_length and b > a]
        out.append(spans)
    return out[0] if squeezed else out


class ZScoreDetector(nn.Module):
    """Flag samples whose step exceeds ``threshold`` running deviations.

    Args:
        window: samples in the running statistic.
        threshold: deviations above which a step counts as an event.
        min_gap: suppress further events within this many samples of one already
            flagged. A single switching transient otherwise fires on every sample
            of its own edge and reports one event as a dozen.
        min_delta: absolute change floor in the signal's own units, so sensor noise
            on an idle channel does not register as appliance activity.

    Example:
        >>> watts = torch.cat([torch.zeros(60), torch.full((60,), 2000.), torch.zeros(60)])
        >>> detector = nf.nn.ZScoreDetector(window=10, threshold=3.0, min_gap=5, min_delta=100.)
        >>> detector
        ZScoreDetector(window=10, threshold=3.0, min_gap=5, min_delta=100.0)
        >>> int(detector(watts).sum())
        0
    """

    def __init__(
        self,
        window: int = 32,
        threshold: float = 4.0,
        min_gap: int = 8,
        min_delta: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.threshold = threshold
        self.min_gap = min_gap
        self.min_delta = min_delta
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Flag samples that step away from their local level.

        Args:
            x: ``(T,)`` or ``(B, T)`` -- a power envelope, usually.

        Returns:
            A boolean mask of the same shape, true where an event was detected.
            A mask rather than indices, so the output shape does not depend on the
            input's contents and a batch cannot go ragged.
        """
        x, squeezed = _as_batched(x)
        mean, deviation = _running_stats(x, self.window)
        score = (x - mean).abs() / deviation.clamp_min(self.eps)
        events = (score > self.threshold) & ((x - mean).abs() > self.min_delta)
        events = _suppress(events, self.min_gap)
        return events.squeeze(0) if squeezed else events

    def extra_repr(self) -> str:
        return (
            f"window={self.window}, threshold={self.threshold}, "
            f"min_gap={self.min_gap}, min_delta={self.min_delta}"
        )


class CusumDetector(nn.Module):
    """Two-sided cumulative sum detector.

    Accumulates signed deviation from a running mean and fires when either
    accumulator crosses ``threshold``, then resets. Because it integrates, it
    responds to a change too gradual to clear a z-score at any single sample.

    Args:
        window: samples in the running mean the deviation is measured against.
        threshold: accumulated deviation that triggers an event, in signal units.
        drift: dead band subtracted from each deviation, so noise does not
            integrate into a false alarm over a long quiet stretch.

    Example:
        >>> watts = torch.cat([torch.zeros(60), torch.full((60,), 2000.), torch.zeros(60)])
        >>> detector = nf.nn.CusumDetector(window=10, threshold=400., drift=25.)
        >>> int(detector(watts).sum())
        16
    """

    def __init__(
        self,
        window: int = 64,
        threshold: float = 50.0,
        drift: float = 1.0,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.threshold = threshold
        self.drift = drift

    def forward(self, x: Tensor) -> Tensor:
        """Flag samples where accumulated deviation crosses the threshold.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            A boolean mask of the same shape. Because the accumulator integrates,
            this fires on changes too gradual for any single sample to look
            anomalous -- a compressor ramp rather than a kettle edge.
        """
        x, squeezed = _as_batched(x)
        mean, _ = _running_stats(x, self.window)
        deviation = x - mean

        # The accumulator resets on every trigger, which is inherently sequential;
        # the running mean above is the part that vectorises.
        batch, length = x.shape
        events = torch.zeros_like(x, dtype=torch.bool)
        up = torch.zeros(batch, dtype=x.dtype, device=x.device)
        down = torch.zeros_like(up)

        for t in range(length):
            step = deviation[:, t]
            up = (up + step - self.drift).clamp_min(0)
            down = (down - step - self.drift).clamp_min(0)
            fired = (up > self.threshold) | (down > self.threshold)
            events[:, t] = fired
            up = torch.where(fired, torch.zeros_like(up), up)
            down = torch.where(fired, torch.zeros_like(down), down)

        return events.squeeze(0) if squeezed else events

    def extra_repr(self) -> str:
        return f"window={self.window}, threshold={self.threshold}, drift={self.drift}"


def _suppress(events: Tensor, min_gap: int) -> Tensor:
    """Keep the first event of each burst, drop the rest within ``min_gap``."""
    if min_gap <= 1:
        return events
    kept = torch.zeros_like(events)
    for b in range(events.shape[0]):
        last = -min_gap
        for t in events[b].nonzero().flatten().tolist():
            if t - last >= min_gap:
                kept[b, t] = True
                last = t
    return kept


class GLRDetector(nn.Module):
    """Generalised likelihood ratio between the windows either side of a sample.

    The classical formulation, and the strongest of the three detectors Anderson
    et al. (2012) compare on BLUED. Where a z-score asks whether *this sample* is
    unusual against recent history, this asks whether the ``pre`` samples before
    a point and the ``post`` samples after it are better explained by one mean or
    by two. That is a question about a *step*, which is what an appliance
    switching actually is, and it is why the statistic is insensitive to a single
    spike: one outlier barely moves either window's mean.

    Under a Gaussian model with a shared variance the log-ratio reduces to

    .. math:: \\frac{n_1 n_2}{n_1 + n_2} \\cdot
              \\frac{(\\mu_{post} - \\mu_{pre})^2}{\\sigma^2}

    so the statistic is a squared mean difference, weighted by how much evidence
    each side carries and normalised by the local noise.

    Args:
        pre: samples before the candidate point.
        post: samples after it.
        threshold: statistic above which a step counts as an event.
        min_gap: suppress further events within this many samples of one flagged.
        min_delta: absolute change floor in the signal's own units.

    Example:
        >>> watts = torch.cat([torch.zeros(60), torch.full((60,), 2000.)])
        >>> detector = nf.nn.GLRDetector(pre=16, post=16, threshold=50., min_gap=8)
        >>> int(detector(watts).sum())
        1
    """

    def __init__(
        self,
        pre: int = 32,
        post: int = 32,
        threshold: float = 100.0,
        min_gap: int = 8,
        min_delta: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if pre < 2 or post < 2:
            raise ValueError(f"pre and post must be >= 2, got {pre} and {post}")
        self.pre = pre
        self.post = post
        self.threshold = threshold
        self.min_gap = min_gap
        self.min_delta = min_delta
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Flag samples where a two-mean model beats a one-mean model.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            A boolean mask of the same shape.
        """
        x, squeezed = _as_batched(x)
        before, after = _window_stats(x, self.pre, self.post)
        mean_pre, var_pre = before
        mean_post, var_post = after

        # One variance for both sides: the null hypothesis is that they share it,
        # and estimating two lets a step inflate its own denominator.
        pooled = (self.pre * var_pre + self.post * var_post) / (self.pre + self.post)
        weight = (self.pre * self.post) / (self.pre + self.post)
        statistic = weight * (mean_post - mean_pre).pow(2) / pooled.clamp_min(self.eps)

        events = (statistic > self.threshold) & ((mean_post - mean_pre).abs() > self.min_delta)
        events = _suppress(events, self.min_gap)
        return events.squeeze(0) if squeezed else events

    def extra_repr(self) -> str:
        return (
            f"pre={self.pre}, post={self.post}, threshold={self.threshold}, "
            f"min_gap={self.min_gap}, min_delta={self.min_delta}"
        )


class GoodnessOfFitDetector(nn.Module):
    """Chi-squared test of the samples after a point against the model before it.

    The third of Anderson et al.'s trio. It differs from :class:`GLRDetector` in
    what counts as a change: the GLR only sees a shift in *mean*, while this sums
    standardised squared residuals and therefore also fires when the level holds
    but the variability changes. A fan stepping between speeds and a motor
    starting to hunt both look like events here and neither does to a mean test.

    The cost is that noise which is merely bursty registers too, which is what
    ``min_delta`` is for.

    Args:
        pre: samples used to fit the reference mean and variance.
        post: samples tested against it.
        threshold: chi-squared statistic above which a change counts.
        min_gap: suppress further events within this many samples.
        min_delta: absolute change floor in the signal's own units.

    Example:
        >>> watts = torch.cat([torch.zeros(60), torch.full((60,), 2000.)])
        >>> detector = nf.nn.GoodnessOfFitDetector(
        ...     pre=16, post=16, threshold=200., min_gap=32)
        >>> int(detector(watts).sum())
        1
    """

    def __init__(
        self,
        pre: int = 32,
        post: int = 32,
        threshold: float = 200.0,
        min_gap: int = 8,
        min_delta: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if pre < 2 or post < 2:
            raise ValueError(f"pre and post must be >= 2, got {pre} and {post}")
        self.pre = pre
        self.post = post
        self.threshold = threshold
        self.min_gap = min_gap
        self.min_delta = min_delta
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Flag samples where the following window does not fit the preceding one.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            A boolean mask of the same shape.
        """
        x, squeezed = _as_batched(x)
        before, after = _window_stats(x, self.pre, self.post)
        mean_pre, var_pre = before
        mean_post, var_post = after

        # Sum of standardised squared residuals of the post window under the
        # pre window's model: n * ((mu_post - mu_pre)^2 + var_post) / var_pre.
        residual = (mean_post - mean_pre).pow(2) + var_post
        statistic = self.post * residual / var_pre.clamp_min(self.eps)

        events = (statistic > self.threshold) & ((mean_post - mean_pre).abs() > self.min_delta)
        events = _suppress(events, self.min_gap)
        return events.squeeze(0) if squeezed else events

    def extra_repr(self) -> str:
        return (
            f"pre={self.pre}, post={self.post}, threshold={self.threshold}, "
            f"min_gap={self.min_gap}, min_delta={self.min_delta}"
        )


class AdaptiveThresholdDetector(nn.Module):
    """A step detector whose threshold follows the local noise level.

    Jin et al. (2011) make the point this class exists for: a fixed threshold is
    a statement about one house. Tuned on a home idling at 300 W it fires
    constantly on one idling at 3 kW, because the noise floor of a mains signal
    scales with what is already running. Every detector above inherits that
    problem through its ``threshold``.

    Here the bar is ``base + scale * sigma``, where ``sigma`` is the running
    deviation over ``window`` samples. On a quiet channel that is ``base``; under
    load it rises with the noise, so the same configuration transfers between
    houses and between datasets.

    Args:
        window: samples in the running statistic.
        base: floor of the threshold, in the signal's own units. Keeps the bar
            from collapsing to zero on a perfectly flat stretch.
        scale: how many running deviations to add to the floor.
        min_gap: suppress further events within this many samples.

    Example:
        >>> quiet = torch.cat([torch.zeros(60), torch.full((60,), 500.)])
        >>> busy = quiet + 3000.
        >>> detector = nf.nn.AdaptiveThresholdDetector(
        ...     window=16, base=50., scale=4., min_gap=32)
        >>> int(detector(quiet).sum()), int(detector(busy).sum())
        (1, 1)
    """

    def __init__(
        self,
        window: int = 32,
        base: float = 20.0,
        scale: float = 4.0,
        min_gap: int = 8,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.base = base
        self.scale = scale
        self.min_gap = min_gap
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Flag steps that clear a bar set by the local noise.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            A boolean mask of the same shape.
        """
        x, squeezed = _as_batched(x)
        before, after = _window_stats(x, self.window, self.window)
        mean_pre, var_pre = before
        mean_post, _ = after

        step = (mean_post - mean_pre).abs()
        bar = self.base + self.scale * var_pre.clamp_min(self.eps).sqrt()
        events = _suppress(step > bar, self.min_gap)
        return events.squeeze(0) if squeezed else events

    def extra_repr(self) -> str:
        return f"window={self.window}, base={self.base}, scale={self.scale}, min_gap={self.min_gap}"


class MultivariateDetector(nn.Module):
    """Run a detector over several measured quantities and combine the verdicts.

    Houidi et al. (2019) observe that a transition invisible in one quantity is
    often obvious in another: a power-factor-correcting load can switch with
    almost no change in active power while reactive power steps hard. Detecting
    on ``p`` alone throws that away, and a store here already carries ``v``,
    ``i`` and ``p`` on one clock.

    Args:
        detector: the per-channel detector to apply. Shared, not copied, so its
            parameters are the same on every channel.
        rule: how to combine. ``"any"`` fires when any channel does -- most
            sensitive, most false alarms. ``"all"`` demands unanimity. ``"vote"``
            fires on a majority, which is the middle and usually the right one.
        align: samples within which two channels' events count as the same event.
            Channels do not fire on the same sample: a reactive step leads an
            active one through the meter's own filtering.

    Example:
        >>> active = torch.cat([torch.zeros(60), torch.full((60,), 2000.)])
        >>> reactive = torch.cat([torch.zeros(62), torch.full((58,), 800.)])
        >>> channels = torch.stack([active, reactive])
        >>> detector = nf.nn.MultivariateDetector(
        ...     nf.nn.GLRDetector(pre=16, post=16, threshold=50., min_gap=32),
        ...     rule="any", align=8)
        >>> int(detector(channels).sum())
        1
    """

    def __init__(self, detector: nn.Module, rule: str = "vote", align: int = 4) -> None:
        super().__init__()
        if rule not in ("any", "all", "vote"):
            raise ValueError(f"rule must be any, all or vote, got {rule!r}")
        self.detector = detector
        self.rule = rule
        self.align = align

    def forward(self, x: Tensor) -> Tensor:
        """Detect on every channel, then combine.

        Args:
            x: ``(C, T)`` or ``(B, C, T)`` -- one row per measured quantity.

        Returns:
            A boolean mask of shape ``(T,)`` or ``(B, T)``: one verdict per
            sample, not one per channel.
        """
        squeezed = x.ndim == 2
        batched = x.unsqueeze(0) if squeezed else x
        batch, channels, length = batched.shape

        fired = self.detector(batched.reshape(batch * channels, length))
        fired = fired.reshape(batch, channels, length)

        # Widen each channel's events before combining, or "all" and "vote" would
        # demand a coincidence the instruments never produce.
        if self.align > 0:
            width = 2 * self.align + 1
            fired = (
                torch.nn.functional.max_pool1d(
                    fired.to(x.dtype), width, stride=1, padding=self.align
                )
                > 0.5
            )

        votes = fired.sum(dim=1)
        if self.rule == "any":
            combined = votes > 0
        elif self.rule == "all":
            combined = votes == channels
        else:
            combined = votes * 2 > channels

        # Widening turned each agreement into a run; take its leading edge, or
        # suppression would keep one event every `min_gap` samples inside it.
        combined = _leading_edge(combined)
        combined = _suppress(combined, self.align * 2 + 1)
        return combined.squeeze(0) if squeezed else combined

    def extra_repr(self) -> str:
        return f"rule={self.rule!r}, align={self.align}"


class ActiveSectionDetector(nn.Module):
    """Segment the signal into steady states and the active sections between them.

    Wild et al. (2015) argue that an instant is the wrong output. A kettle is an
    edge, but a washing machine is minutes of varying draw, and asking an edge
    detector for its "event time" gets an arbitrary sample inside the ramp, or a
    dozen events for one activation. Their detector returns *intervals* instead:
    the signal is either in a steady state or in an active section, and the
    boundaries are what you extract features between.

    A sample is active when the local deviation exceeds ``threshold``; runs
    shorter than ``min_steady`` are absorbed into their neighbours so that a
    momentary settle inside one ramp does not split it in two.

    Args:
        window: samples in the running deviation.
        threshold: deviation above which the signal counts as active.
        min_steady: shortest run, in samples, that may be called a steady state.

    Example:
        >>> ramp = torch.cat([torch.zeros(40), torch.linspace(0, 2000, 40),
        ...                   torch.full((40,), 2000.)])
        >>> detector = nf.nn.ActiveSectionDetector(window=8, threshold=20., min_steady=5)
        >>> spans = detector.sections(ramp)
        >>> len(spans) >= 1 and spans[0][1] > spans[0][0]
        True
    """

    def __init__(self, window: int = 16, threshold: float = 20.0, min_steady: int = 8) -> None:
        super().__init__()
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.threshold = threshold
        self.min_steady = min_steady

    def forward(self, x: Tensor) -> Tensor:
        """Mark the samples that belong to an active section.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            A boolean mask of the same shape -- true inside an active section.
            Unlike the edge detectors this is not sparse: it covers whole
            transitions rather than marking their midpoints.
        """
        x, squeezed = _as_batched(x)
        _, deviation = _running_stats(x, self.window)
        active = deviation > self.threshold
        active = _close_short_runs(active, self.min_steady)
        return active.squeeze(0) if squeezed else active

    def sections(self, x: Tensor) -> list[tuple[int, int]] | list[list[tuple[int, int]]]:
        """The active sections as ``[start, stop)`` spans.

        Args:
            x: ``(T,)`` or ``(B, T)``.

        Returns:
            One list of spans per item, or a single list for unbatched input.
        """
        mask, squeezed = _as_batched(self.forward(x))
        out = [_runs_of(row) for row in mask]
        return out[0] if squeezed else out

    def extra_repr(self) -> str:
        return f"window={self.window}, threshold={self.threshold}, min_steady={self.min_steady}"


def _window_stats(
    x: Tensor, pre: int, post: int
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Mean and variance of the ``pre`` samples before and ``post`` after each point."""

    def stats(padded: Tensor, width: int) -> tuple[Tensor, Tensor]:
        mean = torch.nn.functional.avg_pool1d(padded, width, stride=1).squeeze(1)
        square = torch.nn.functional.avg_pool1d(padded.pow(2), width, stride=1).squeeze(1)
        return mean, (square - mean.pow(2)).clamp_min(0)

    body = x.unsqueeze(1)
    # Left window ends at the sample before; right window starts at the sample.
    left = torch.nn.functional.pad(body, (pre, 0), mode="replicate")[..., :-1]
    right = torch.nn.functional.pad(body, (0, post), mode="replicate")[..., 1:]
    return stats(left, pre), stats(right, post)


def _leading_edge(mask: Tensor) -> Tensor:
    """Keep only the first sample of each run of ``True``."""
    previous = torch.nn.functional.pad(mask[:, :-1], (1, 0), value=False)
    return mask & ~previous


def _close_short_runs(mask: Tensor, min_length: int) -> Tensor:
    """Absorb runs shorter than ``min_length`` into whatever surrounds them."""
    if min_length <= 1:
        return mask
    out = mask.clone()
    for b in range(mask.shape[0]):
        row = out[b]
        start = 0
        for stop in range(1, row.numel() + 1):
            if stop == row.numel() or row[stop] != row[start]:
                if stop - start < min_length and start > 0:
                    row[start:stop] = row[start - 1]
                start = stop
    return out


def _runs_of(row: Tensor) -> list[tuple[int, int]]:
    """``[start, stop)`` spans where a boolean row is true."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for t, on in enumerate(row.tolist()):
        if on and start is None:
            start = t
        elif not on and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, int(row.numel())))
    return spans
