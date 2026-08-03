"""NILM metrics, in four families reported separately.

Mixing detection quality and power accuracy into one number is why the
predecessor's results were hard to interpret: a model could get every appliance
right and every wattage wrong and still score well, or the reverse, and the single
figure could not tell you which.

**Detection** -- per-appliance and macro F1, and MCC. Do we know *what* is on.

**Power** -- MAE, NDE and SAE, the quantities the NILM literature reports, so a
number here is comparable with published work rather than only with itself.

**Joint** -- ``ModifiedF1`` and ``ModifiedJaccard`` from
``legacy/metrics/multioutput.py``, kept because they are a real contribution: a
detection counts only when the power estimate is also within tolerance. Ported
with the defects fixed -- an epsilon in every denominator (the base class divided
by ``Y_pred.sum(1)`` with none, so an all-off batch returned NaN), all-off windows
excluded from TECA rather than dividing by zero, and no rescaling by
``Y_true.sum(1)``.

**Open set** -- AUROC for known versus unknown, and expected calibration error.
A model that is confidently wrong about an appliance it has never seen is the
failure mode ``open_set_note.tex`` is about, and neither F1 nor MAE detects it.

Every metric is a ``torchmetrics.Metric``, so accumulation across batches and
across DDP ranks is handled and ``MetricCollection`` composes them.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from torchmetrics import Metric
except ImportError as exc:  # pragma: no cover - exercised by the extras matrix
    raise ImportError("nilmframe.eval needs torchmetrics: pip install 'nilmframe[eval]'") from exc

__all__ = [
    "SAE",
    "TECA",
    "CalibrationError",
    "DetectionF1",
    "MatthewsCorrCoef",
    "MeanAbsoluteError",
    "ModifiedF1",
    "ModifiedJaccard",
    "NormalisedDisaggregationError",
    "UnknownAUROC",
    "default_collection",
]

_EPS = 1e-9


def _as_bool(x: Tensor) -> Tensor:
    return x > 0.5 if x.dtype.is_floating_point else x.to(torch.bool)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


class DetectionF1(Metric):
    """Per-appliance F1, plus macro and micro averages.

    Args:
        n_appliances: label-space size.

    Example:
        >>> from nilmframe.eval import DetectionF1
        >>> truth = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        >>> metric = DetectionF1(3)
        >>> metric.update(truth, truth)
        >>> round(float(metric.compute()['f1_macro']), 3)
        1.0
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, n_appliances: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_appliances = n_appliances
        for name in ("tp", "fp", "fn"):
            self.add_state(name, default=torch.zeros(n_appliances), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred, target = _as_bool(pred), _as_bool(target)
        self.tp += (pred & target).sum(0).float()
        self.fp += (pred & ~target).sum(0).float()
        self.fn += (~pred & target).sum(0).float()

    def compute(self) -> dict[str, Tensor]:
        per_appliance = 2 * self.tp / (2 * self.tp + self.fp + self.fn + _EPS)
        # Only average over appliances that actually occur; a class absent from the
        # evaluation set otherwise contributes a spurious zero.
        seen = (self.tp + self.fn) > 0
        macro = per_appliance[seen].mean() if seen.any() else torch.zeros((), device=self.tp.device)
        tp, fp, fn = self.tp.sum(), self.fp.sum(), self.fn.sum()
        return {
            "f1_macro": macro,
            "f1_micro": 2 * tp / (2 * tp + fp + fn + _EPS),
            "f1_per_appliance": per_appliance,
        }


class MatthewsCorrCoef(Metric):
    """Micro-averaged MCC: robust to the heavy class imbalance NILM targets have.

    Example:
        >>> from nilmframe.eval import MatthewsCorrCoef
        >>> truth = torch.tensor([[1., 0.], [0., 1.]])
        >>> metric = MatthewsCorrCoef()
        >>> metric.update(truth, truth)
        >>> round(float(metric.compute()), 3)
        1.0
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for name in ("tp", "fp", "fn", "tn"):
            self.add_state(name, default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate one batch.

        Args:
            pred: ``(B, K)`` predicted presence, boolean or a probability.
            target: ``(B, K)`` true presence.
        """
        pred, target = _as_bool(pred), _as_bool(target)
        self.tp += (pred & target).sum().float()
        self.fp += (pred & ~target).sum().float()
        self.fn += (~pred & target).sum().float()
        self.tn += (~pred & ~target).sum().float()

    def compute(self) -> Tensor:
        """Matthews correlation in ``[-1, 1]``. Zero is chance, one is perfect.

        Unlike F1 it accounts for true negatives, so an appliance that is off
        almost always cannot be scored well by simply never predicting it.
        """
        num = self.tp * self.tn - self.fp * self.fn
        den = torch.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        )
        return num / den.clamp_min(_EPS)


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


class MeanAbsoluteError(Metric):
    """MAE in watts over the entries whose true power is known.

    Example:
        >>> from nilmframe.eval import MeanAbsoluteError
        >>> metric = MeanAbsoluteError()
        >>> metric.update(torch.tensor([[100., 0.]]), torch.tensor([[110., 0.]]))
        >>> round(float(metric.compute()), 2)
        5.0
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("error", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        """Accumulate one batch.

        Args:
            pred: ``(B, K)`` predicted watts.
            target: ``(B, K)`` true watts.
            mask: ``(B, K)``, true where the target is *known*. Entries that are
                masked out contribute nothing -- an aggregate annotated only with
                on/off times has no per-appliance watts, and scoring against an
                invented zero would flatter the model.
        """
        mask = torch.ones_like(target, dtype=torch.bool) if mask is None else _as_bool(mask)
        self.error += ((pred - target).abs() * mask).sum()
        self.count += mask.sum()

    def compute(self) -> Tensor:
        """Mean absolute error in watts over everything accumulated so far."""
        return self.error / self.count.clamp_min(1)


class NormalisedDisaggregationError(Metric):
    """NDE: total absolute error over total true energy.

    Scale-free, so it is comparable across appliances of very different sizes -- a
    50 W error on a fridge and on an oven are not the same mistake.

    Example:
        >>> from nilmframe.eval import NormalisedDisaggregationError
        >>> metric = NormalisedDisaggregationError()
        >>> metric.update(torch.tensor([[90.]]), torch.tensor([[100.]]))
        >>> round(float(metric.compute()), 3)
        0.1
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("error", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        """Accumulate one batch.

        Args:
            pred: ``(B, K)`` predicted watts.
            target: ``(B, K)`` true watts.
            mask: ``(B, K)``, true where the target is known.
        """
        mask = torch.ones_like(target, dtype=torch.bool) if mask is None else _as_bool(mask)
        self.error += ((pred - target).abs() * mask).sum()
        self.total += (target.abs() * mask).sum()

    def compute(self) -> Tensor:
        """Total absolute error over total true energy. Zero is perfect; 1.0 is
        what predicting nothing at all scores."""
        return self.error / self.total.clamp_min(_EPS)


class SAE(Metric):
    """Signal aggregate error: relative error of *total* energy per appliance.

    A model can place energy in the wrong windows and still get the daily total
    right; SAE is what tells those two failure modes apart.

    Example:
        >>> from nilmframe.eval import SAE
        >>> metric = SAE(1)
        >>> metric.update(torch.tensor([[0.], [200.]]), torch.tensor([[200.], [0.]]))
        >>> round(float(metric.compute()), 5)
        0.0
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_appliances: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("pred", default=torch.zeros(n_appliances), dist_reduce_fx="sum")
        self.add_state("target", default=torch.zeros(n_appliances), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        """Accumulate per-appliance energy totals.

        Args:
            pred: ``(B, K)`` predicted watts.
            target: ``(B, K)`` true watts.
            mask: ``(B, K)``, true where the target is known.
        """
        mask = torch.ones_like(target, dtype=torch.bool) if mask is None else _as_bool(mask)
        self.pred += (pred * mask).sum(0)
        self.target += (target * mask).sum(0)

    def compute(self) -> Tensor:
        """Mean relative error of total energy, over appliances that drew any.

        Returns:
            Zero when the totals are right, whether or not the energy landed in
            the right windows -- which is exactly what separates it from
            :class:`MeanAbsoluteError`.
        """
        seen = self.target.abs() > _EPS
        if not seen.any():
            return torch.zeros((), device=self.pred.device)
        return ((self.pred - self.target).abs() / self.target.abs().clamp_min(_EPS))[seen].mean()


# --------------------------------------------------------------------------- #
# Joint -- ported from legacy/metrics/multioutput.py
# --------------------------------------------------------------------------- #


class TECA(Metric):
    """Total energy correctly assigned.

    ``1 - sum|y - y_hat| / (2 * sum y)``, averaged over windows. Windows with no
    load at all are skipped: the legacy version divided by ``Y_true.sum(1)``
    without guarding, so an all-off window produced a division by zero.

    Example:
        >>> from nilmframe.eval import TECA
        >>> metric = TECA()
        >>> metric.update(torch.tensor([[100., 50.]]), torch.tensor([[100., 50.]]))
        >>> round(float(metric.compute()), 3)
        1.0
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("score", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("count", default=torch.zeros(()), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate one batch.

        Args:
            pred: ``(B, K)`` predicted watts.
            target: ``(B, K)`` true watts. Windows whose total is zero are
                skipped rather than dividing by it.
        """
        total = target.sum(-1)
        valid = total > _EPS
        if not valid.any():
            return
        score = 1 - (pred - target).abs().sum(-1)[valid] / (2 * total[valid])
        self.score += score.sum()
        self.count += valid.sum()

    def compute(self) -> Tensor:
        """Mean fraction of energy assigned to the right appliance. 1.0 is perfect."""
        return self.score / self.count.clamp_min(1)


class _ToleranceMetric(Metric):
    """Shared machinery for the two tolerance-aware metrics.

    A true positive is *accurate* when the power error is within tolerance and
    *inaccurate* otherwise; an inaccurate one counts against the score without
    counting as a miss.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, delta: float, relative: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.relative = relative
        for name in ("atp", "itp", "fp", "fn"):
            self.add_state(name, default=torch.zeros(()), dist_reduce_fx="sum")

    def update(
        self,
        presence_pred: Tensor,
        power_pred: Tensor,
        presence_true: Tensor,
        power_true: Tensor,
    ) -> None:
        on_pred, on_true = _as_bool(presence_pred), _as_bool(presence_true)
        if self.relative:
            error = (power_pred - power_true).abs() / power_true.abs().clamp_min(_EPS)
        else:
            error = (power_pred - power_true).abs()
        within = error <= self.delta

        hit = on_pred & on_true
        self.atp += (hit & within).sum().float()
        self.itp += (hit & ~within).sum().float()
        self.fp += (on_pred & ~on_true).sum().float()
        self.fn += (~on_pred & on_true).sum().float()


class ModifiedF1(_ToleranceMetric):
    """F1 where a detection counts only if the power estimate is close enough.

    Args:
        delta: relative power tolerance, e.g. ``0.2`` for 20%.

    Example:
        >>> from nilmframe.eval import ModifiedF1
        >>> presence, truth = torch.tensor([[1.]]), torch.tensor([[100.]])
        >>> close = ModifiedF1(delta=0.2)
        >>> close.update(presence, torch.tensor([[110.]]), presence, truth)
        >>> round(float(close.compute()), 3)
        1.0
        >>> far = ModifiedF1(delta=0.2)
        >>> far.update(presence, torch.tensor([[900.]]), presence, truth)
        >>> round(float(far.compute()), 3)
        0.0
    """

    def __init__(self, delta: float = 0.2, **kwargs) -> None:
        super().__init__(delta=delta, relative=True, **kwargs)

    def compute(self) -> Tensor:
        return self.atp / (self.atp + self.itp + 0.5 * (self.fp + self.fn) + _EPS)


class ModifiedJaccard(_ToleranceMetric):
    """Jaccard with an absolute power tolerance.

    Args:
        delta: absolute tolerance in watts.

    Example:
        >>> from nilmframe.eval import ModifiedJaccard
        >>> presence, truth = torch.tensor([[1.]]), torch.tensor([[1000.]])
        >>> metric = ModifiedJaccard(delta=20.0)
        >>> metric.update(presence, torch.tensor([[1015.]]), presence, truth)
        >>> round(float(metric.compute()), 3)
        1.0
    """

    def __init__(self, delta: float = 20.0, **kwargs) -> None:
        super().__init__(delta=delta, relative=False, **kwargs)

    def compute(self) -> Tensor:
        return self.atp / (self.atp + self.itp + self.fp + self.fn + _EPS)


# --------------------------------------------------------------------------- #
# Open set
# --------------------------------------------------------------------------- #


class UnknownAUROC(Metric):
    """AUROC for separating windows containing an unseen appliance from the rest.

    Computed exactly from the rank statistic rather than by sampling, so a small
    evaluation set does not get a noisy answer.

    Example:
        >>> from nilmframe.eval import UnknownAUROC
        >>> metric = UnknownAUROC()
        >>> metric.update(torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1, 1, 0, 0]))
        >>> round(float(metric.compute()), 3)
        1.0
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, score: Tensor, is_unknown: Tensor) -> None:
        """Accumulate one batch.

        Args:
            score: ``(B,)`` the model's open-set score, higher meaning
                "more likely out of vocabulary".
            is_unknown: ``(B,)`` whether the window really contained a held-out
                appliance.
        """
        self.scores.append(score.detach().flatten())
        self.labels.append(_as_bool(is_unknown).flatten())

    def compute(self) -> Tensor:
        scores = torch.cat(self.scores) if isinstance(self.scores, list) else self.scores
        labels = torch.cat(self.labels) if isinstance(self.labels, list) else self.labels
        """Area under the ROC curve, computed exactly from the rank statistic.

        Returns:
            1.0 when every unknown window scores above every known one, 0.5 for
            chance, and NaN when one of the two classes is absent -- undefined
            rather than zero, since zero would read as a real result.
        """
        positive, negative = labels.sum(), (~labels).sum()
        if positive == 0 or negative == 0:
            # Undefined rather than zero: one class is missing entirely.
            return torch.full((), float("nan"), device=scores.device)
        order = scores.argsort()
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float)
        return (ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative)


class CalibrationError(Metric):
    """Expected calibration error of the presence probabilities.

    Args:
        n_bins: equal-width confidence bins.

    Example:
        >>> from nilmframe.eval import CalibrationError
        >>> metric = CalibrationError(n_bins=10)
        >>> metric.update(torch.tensor([1., 1., 0., 0.]), torch.tensor([1., 1., 0., 0.]))
        >>> round(float(metric.compute()), 4)
        0.0
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, n_bins: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_bins = n_bins
        for name in ("correct", "confidence", "count"):
            self.add_state(name, default=torch.zeros(n_bins), dist_reduce_fx="sum")

    def update(self, probability: Tensor, target: Tensor) -> None:
        probability = probability.flatten()
        target = _as_bool(target).flatten().float()
        # Confidence in the predicted class, so a confident "off" counts too.
        predicted = (probability > 0.5).float()
        confidence = torch.where(predicted > 0, probability, 1 - probability)
        correct = (predicted == target).float()

        idx = (confidence * self.n_bins).long().clamp(0, self.n_bins - 1)
        self.count += torch.bincount(idx, minlength=self.n_bins).float()
        self.confidence += torch.bincount(idx, weights=confidence, minlength=self.n_bins).float()
        self.correct += torch.bincount(idx, weights=correct, minlength=self.n_bins).float()

    def compute(self) -> Tensor:
        total = self.count.sum().clamp_min(1)
        seen = self.count > 0
        gap = (self.confidence[seen] - self.correct[seen]).abs()
        return (gap / self.count[seen] * self.count[seen]).sum() / total


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def default_collection(
    n_appliances: int,
    *,
    modified_f1_delta: float = 0.2,
    modified_jaccard_delta: float = 20.0,
    open_set: bool = False,
) -> dict[str, Metric]:
    """The four families, ready to update.

    Returned as a plain dict rather than a ``MetricCollection`` because the metrics
    take different argument signatures -- detection wants booleans, power wants
    watts and a mask, the joint ones want all four. See
    :class:`~nilmframe.eval.evaluator.Evaluator`, which routes them.

    Example:
        >>> from nilmframe.eval import default_collection
        >>> for name in sorted(default_collection(n_appliances=4)):
        ...     print(name)
        ...
        calibration
        detection
        mae
        mcc
        modified_f1
        modified_jaccard
        nde
        sae
        teca
    """
    metrics: dict[str, Metric] = {
        "detection": DetectionF1(n_appliances),
        "mcc": MatthewsCorrCoef(),
        "mae": MeanAbsoluteError(),
        "nde": NormalisedDisaggregationError(),
        "sae": SAE(n_appliances),
        "teca": TECA(),
        "modified_f1": ModifiedF1(modified_f1_delta),
        "modified_jaccard": ModifiedJaccard(modified_jaccard_delta),
        "calibration": CalibrationError(),
    }
    if open_set:
        metrics["unknown_auroc"] = UnknownAUROC()
    return metrics
