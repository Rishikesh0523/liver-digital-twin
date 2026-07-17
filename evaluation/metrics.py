"""Predictive-accuracy and constraint metrics, per-field and aggregate.

Kept deliberately simple and honest: MSE/MAE per field, a constraint-violation
rate (should be 0 for the constrained models — reported anyway as a tripwire),
and error-vs-horizon so we can see accumulation, not just a single headline number.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data.schema import FIELD_NAMES, STATE_DIM
from ..models.constraints import constraint_violations


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def mse_per_field(pred, target) -> dict[str, float]:
    p, t = _to_np(pred), _to_np(target)
    err = ((p - t) ** 2).reshape(-1, STATE_DIM).mean(0)
    return {FIELD_NAMES[i]: float(err[i]) for i in range(STATE_DIM)}


def mae_per_field(pred, target) -> dict[str, float]:
    p, t = _to_np(pred), _to_np(target)
    err = np.abs(p - t).reshape(-1, STATE_DIM).mean(0)
    return {FIELD_NAMES[i]: float(err[i]) for i in range(STATE_DIM)}


def overall_mse(pred, target) -> float:
    p, t = _to_np(pred), _to_np(target)
    return float(((p - t) ** 2).mean())


def error_vs_horizon(pred, target) -> list[float]:
    """Mean MSE at each future step (pred/target are (B, K, 8))."""
    p, t = _to_np(pred), _to_np(target)
    return [float(((p[:, k] - t[:, k]) ** 2).mean()) for k in range(p.shape[1])]


def cumulative_error(pred, target) -> float:
    """Integrated (summed) per-step MSE over the horizon."""
    return float(np.sum(error_vs_horizon(pred, target)))


def constraint_violation_rate(states: torch.Tensor, ercp: torch.Tensor) -> dict[str, float]:
    """Fraction of transitions that violate each hard constraint.

    Normalised by the number of transitions so it reads as a rate in [0,1].
    """
    v = constraint_violations(states, ercp)
    B, T, _ = states.shape
    n_trans = B * (T - 1)
    out = {k: float(x) / max(1, n_trans) for k, x in v.items() if k != "total"}
    out["any_rate"] = float(v["total"]) / max(1, n_trans)
    out["_counts"] = {k: int(x) for k, x in v.items()}
    return out


def bounds_violation_rate(states: torch.Tensor) -> float:
    from ..data.schema import upper_bounds
    upper = torch.tensor(upper_bounds(), device=states.device)
    below = (states < -1e-5).float().mean()
    above = (states > upper + 1e-5).float().mean()
    return float(below + above)


@torch.no_grad()
def evaluate_predictor(predict_fn, loader, device) -> dict:
    """Run a model's predictions over a loader and compute the full metric bundle.

    ``predict_fn(batch) -> pred_states (B,K,8)`` abstracts JEPA vs baseline.
    """
    preds, targets, ercps = [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred = predict_fn(batch)
        preds.append(pred.cpu())
        targets.append(batch.future.cpu())
        ercps.append(batch.ercp_future.cpu())
    pred = torch.cat(preds)
    target = torch.cat(targets)
    ercp = torch.cat(ercps)
    return {
        "overall_mse": overall_mse(pred, target),
        "mse_per_field": mse_per_field(pred, target),
        "mae_per_field": mae_per_field(pred, target),
        "error_vs_horizon": error_vs_horizon(pred, target),
        "cumulative_error": cumulative_error(pred, target),
        "constraint_violation_rate": constraint_violation_rate(pred, ercp),
        "bounds_violation_rate": bounds_violation_rate(pred),
        "n_samples": int(pred.shape[0]),
    }
