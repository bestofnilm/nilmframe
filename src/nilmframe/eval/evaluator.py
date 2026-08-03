"""Routing predictions and targets into the metric families.

Each family wants different arguments -- detection takes booleans, power takes
watts and a knownness mask, the tolerance metrics take all four -- so something has
to route them. Doing it here, once, means a training loop never has to remember
which metric wants what, and it means adding a metric does not touch the loop.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from nilmframe.eval.metrics import default_collection

__all__ = ["Evaluator"]


class Evaluator:
    """Accumulate metrics over batches and report them.

    Args:
        n_appliances: label-space size.
        appliances: names, for the per-appliance breakdown.
        open_set: include the open-set metrics.
        **kwargs: forwarded to
            :func:`~nilmframe.eval.metrics.default_collection`.

    Note:
        There is no threshold argument. Deciding what counts as "on" belongs to
        a model's ``predict``, which has already
        applied it by the time an evaluator sees the result. Accepting one here
        would be a second, silently ignored knob.

    Example:
        >>> evaluator = nf.eval.Evaluator(3, ['kettle', 'fridge', 'laptop'])
        >>> prediction = {'presence': torch.tensor([[True, False, True]]),
        ...               'power': torch.tensor([[2000., 0., 40.]]),
        ...               'probability': torch.tensor([[0.9, 0.1, 0.8]])}
        >>> batch = {'presence': torch.tensor([[1., 0., 1.]]),
        ...          'power': torch.tensor([[2000., 0., 40.]])}
        >>> evaluator.update(prediction, batch)
        >>> round(evaluator.compute()['f1_macro'], 3)
        1.0
        >>> print(evaluator.report())
        detection  f1_macro=1.0000  f1_micro=1.0000  mcc=1.0000
        power      mae=0.0000  nde=0.0000  sae=0.0000
        joint      teca=1.0000  modified_f1=1.0000  modified_jaccard=1.0000
        open-set   calibration=0.1333
    """

    def __init__(
        self,
        n_appliances: int,
        appliances: list[str] | None = None,
        *,
        open_set: bool = False,
        **kwargs,
    ) -> None:
        self.n_appliances = n_appliances
        self.appliances = list(appliances) if appliances else [f"a{k}" for k in range(n_appliances)]
        self.metrics = default_collection(n_appliances, open_set=open_set, **kwargs)
        # A metric that never saw data has nothing to say. Reporting its empty
        # state as 0.0 would look like a result, so track what was actually fed.
        self._updated: set[str] = set()

    def to(self, device) -> Evaluator:
        for metric in self.metrics.values():
            metric.to(device)
        return self

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self._updated.clear()

    def update(self, prediction: dict[str, Tensor], batch: dict[str, Any]) -> None:
        """Feed one batch.

        Args:
            prediction: the output of
                a model's ``predict``.
            batch: the dataset batch, for targets.
        """
        presence_pred = prediction["presence"]
        power_pred = prediction["power"]
        probability = prediction.get("probability", presence_pred.float())

        presence_true = batch.get("presence")
        power_true = batch.get("power")
        power_mask = batch.get("power_mask")

        if presence_true is not None:
            self.metrics["detection"].update(presence_pred, presence_true)
            self.metrics["mcc"].update(presence_pred, presence_true)
            self.metrics["calibration"].update(probability, presence_true)
            self._updated |= {"detection", "mcc", "calibration"}

        if power_true is not None:
            self.metrics["mae"].update(power_pred, power_true, power_mask)
            self.metrics["nde"].update(power_pred, power_true, power_mask)
            self.metrics["sae"].update(power_pred, power_true, power_mask)
            self.metrics["teca"].update(power_pred, power_true)
            self._updated |= {"mae", "nde", "sae", "teca"}

            if presence_true is not None:
                for name in ("modified_f1", "modified_jaccard"):
                    self.metrics[name].update(presence_pred, power_pred, presence_true, power_true)
                self._updated |= {"modified_f1", "modified_jaccard"}

        if "unknown_auroc" in self.metrics and "is_unknown" in batch:
            score = prediction.get("unknown_score")
            if score is not None:
                self.metrics["unknown_auroc"].update(score, batch["is_unknown"])
                self._updated.add("unknown_auroc")

    def compute(self) -> dict[str, float]:
        """Flatten every family into one dict of scalars."""
        out: dict[str, float] = {}
        for name, metric in self.metrics.items():
            if name not in self._updated:
                continue
            value = metric.compute()
            if isinstance(value, dict):
                for key, item in value.items():
                    if item.ndim == 0:
                        out[key] = float(item)
                    else:
                        for appliance, scalar in zip(self.appliances, item.tolist(), strict=False):
                            out[f"{key}:{appliance}"] = float(scalar)
            elif torch.is_tensor(value) and value.ndim == 0:
                out[name] = float(value)
        return out

    def report(self) -> str:
        """A human-readable summary, grouped by family."""
        values = self.compute()
        families = {
            "detection": ("f1_macro", "f1_micro", "mcc"),
            "power": ("mae", "nde", "sae"),
            "joint": ("teca", "modified_f1", "modified_jaccard"),
            "open-set": ("unknown_auroc", "calibration"),
        }
        lines = []
        for family, keys in families.items():
            present = [(k, values[k]) for k in keys if k in values]
            if present:
                body = "  ".join(f"{k}={v:.4f}" for k, v in present)
                lines.append(f"{family:<10} {body}")
        return "\n".join(lines)
