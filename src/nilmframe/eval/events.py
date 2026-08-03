"""Event-detection metrics: did we find the transition, and when.

These answer a different question from the rest of :mod:`nilmframe.eval`.
:class:`~nilmframe.eval.DetectionF1` asks *which appliance is on in this window*.
Stage one of an event-based pipeline does not know about appliances at all -- it
claims that something switched at time ``t``, and the only questions are whether
something did, and how close ``t`` was.

That makes the matching the whole problem. A detector firing one sample late is
right; the same detector firing a minute late is wrong; and no detector lands
exactly on the annotator's timestamp. So a detection counts when it falls inside
``tolerance`` samples of a true event, and each true event can be claimed once --
without that second rule a detector that fires continuously scores a perfect
recall.

The matching is greedy by distance, which is what Anderson et al. (2012) use and
what makes the numbers here comparable with the published ones. The alternative
-- optimal bipartite assignment -- differs only when detections are dense enough
to contest the same event, and by then the detector is not usable anyway.

Three numbers are reported rather than one:

:class:`EventF1`
    Precision, recall and F1 over matched events. The headline.
:class:`EventTimingError`
    Mean absolute distance, in samples, between a matched pair. A detector can
    have perfect F1 at a loose tolerance and still be useless for extracting the
    transient, and this is what says so.
:class:`EventCounts`
    Raw true positives, false positives and false negatives. Reported because the
    two failure modes -- missing events and inventing them -- are not
    interchangeable, and F1 hides which one you have.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from torchmetrics import Metric
except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
    raise ImportError("nilmframe.eval needs torchmetrics: pip install 'nilmframe[eval]'") from exc

__all__ = ["EventCounts", "EventF1", "EventTimingError", "match_events"]

_EPS = 1e-9


def _indices(events: Tensor) -> Tensor:
    """Event positions, whether given as a boolean mask or as indices already."""
    if events.dtype == torch.bool:
        return events.nonzero().flatten()
    if events.dtype.is_floating_point:
        return (events > 0.5).nonzero().flatten()
    return events.flatten()


def match_events(
    predicted: Tensor, actual: Tensor, tolerance: int = 8
) -> tuple[Tensor, Tensor, int, int]:
    """Pair detections with true events, nearest first.

    Args:
        predicted: detected events -- a boolean mask over time, or sample indices.
        actual: true events, in the same form.
        tolerance: how many samples a detection may be off by and still count.

    Returns:
        ``(matched_predicted, matched_actual, false_positives, false_negatives)``.
        The first two are index tensors of equal length, paired elementwise.

    Example:
        >>> import torch
        >>> from nilmframe.eval import match_events
        >>> predicted = torch.tensor([10, 51, 200])
        >>> actual = torch.tensor([12, 50])
        >>> hits, truths, extra, missed = match_events(predicted, actual, tolerance=5)
        >>> hits.tolist(), truths.tolist(), extra, missed
        ([10, 51], [12, 50], 1, 0)
    """
    found = _indices(predicted).to(torch.long)
    truth = _indices(actual).to(torch.long)
    if found.numel() == 0 or truth.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=found.device)
        return empty, empty, int(found.numel()), int(truth.numel())

    distance = (found[:, None] - truth[None, :]).abs()
    # Greedy nearest-first: the closest surviving pair is matched, then both are
    # struck out. A true event can be claimed once, or a detector that fires on
    # every sample would recall everything.
    order = torch.argsort(distance.flatten())
    n_truth = truth.numel()
    used_found: set[int] = set()
    used_truth: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for flat in order.tolist():
        row, column = divmod(flat, n_truth)
        if distance[row, column] > tolerance:
            break  # sorted, so everything after is further still
        if row in used_found or column in used_truth:
            continue
        used_found.add(row)
        used_truth.add(column)
        pairs.append((row, column))

    if not pairs:
        empty = torch.empty(0, dtype=torch.long, device=found.device)
        return empty, empty, int(found.numel()), int(truth.numel())

    # Matched greedily by distance, but returned in time order: a caller
    # zipping the two tensors wants them chronological, not by match quality.
    pairs.sort()
    rows = torch.tensor([p[0] for p in pairs], device=found.device)
    columns = torch.tensor([p[1] for p in pairs], device=found.device)
    return (
        found[rows],
        truth[columns],
        int(found.numel() - len(pairs)),
        int(truth.numel() - len(pairs)),
    )


class _EventMetric(Metric):
    """Shared accumulation: match each item, then keep the tallies."""

    higher_is_better = True
    full_state_update = False

    def __init__(self, tolerance: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)
        if tolerance < 0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self.tolerance = tolerance
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("offset", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, predicted: Tensor, actual: Tensor) -> None:
        """Accumulate one item, or a batch of them.

        Args:
            predicted: ``(T,)`` or ``(B, T)`` detections, or index tensors.
            actual: true events in the same form.
        """
        if predicted.ndim > 1 or actual.ndim > 1:
            rows = max(predicted.shape[0], actual.shape[0])
            for row in range(rows):
                self._accumulate(predicted[row], actual[row])
        else:
            self._accumulate(predicted, actual)

    def _accumulate(self, predicted: Tensor, actual: Tensor) -> None:
        hits, truths, extra, missed = match_events(predicted, actual, self.tolerance)
        self.tp += float(hits.numel())
        self.fp += float(extra)
        self.fn += float(missed)
        if hits.numel():
            self.offset += float((hits - truths).abs().sum())


class EventF1(_EventMetric):
    """F1 over matched events, ignoring which appliance caused them.

    Args:
        tolerance: samples a detection may be off by and still count.

    Example:
        >>> import torch
        >>> from nilmframe.eval import EventF1
        >>> metric = EventF1(tolerance=5)
        >>> _ = metric.update(torch.tensor([10, 51, 200]), torch.tensor([12, 50]))
        >>> round(float(metric.compute()), 3)
        0.8
    """

    def compute(self) -> Tensor:
        precision = self.tp / (self.tp + self.fp + _EPS)
        recall = self.tp / (self.tp + self.fn + _EPS)
        return 2 * precision * recall / (precision + recall + _EPS)

    def precision_recall(self) -> tuple[Tensor, Tensor]:
        """The two halves, for when the balance matters more than the average."""
        return (
            self.tp / (self.tp + self.fp + _EPS),
            self.tp / (self.tp + self.fn + _EPS),
        )


class EventTimingError(_EventMetric):
    """Mean absolute distance, in samples, between a detection and its event.

    Only matched pairs contribute -- a missed event has no timing error, it has a
    recall penalty, and folding the two together would let a detector improve this
    number by detecting less.

    Args:
        tolerance: samples a detection may be off by and still count.

    Example:
        >>> import torch
        >>> from nilmframe.eval import EventTimingError
        >>> metric = EventTimingError(tolerance=5)
        >>> _ = metric.update(torch.tensor([10, 51]), torch.tensor([12, 50]))
        >>> float(metric.compute())
        1.5
    """

    higher_is_better = False

    def compute(self) -> Tensor:
        return self.offset / (self.tp + _EPS)


class EventCounts(_EventMetric):
    """Raw true positives, false positives and false negatives.

    Example:
        >>> import torch
        >>> from nilmframe.eval import EventCounts
        >>> metric = EventCounts(tolerance=5)
        >>> _ = metric.update(torch.tensor([10, 51, 200]), torch.tensor([12, 50]))
        >>> {k: int(v) for k, v in metric.compute().items()}
        {'tp': 2, 'fp': 1, 'fn': 0}
    """

    def compute(self) -> dict[str, Tensor]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}
