"""Model integration tests: shapes, the by-construction guarantee end-to-end,
collapse metric behaviour, and a one-step overfit sanity check.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from liver_world_model.data.dataset import make_splits, collate
from liver_world_model.models.jepa import LiverJEPA
from liver_world_model.models.baseline import DirectPredictor
from liver_world_model.models.collapse import compute_effective_dimensionality
from liver_world_model.models.constraints import constraint_violations
from liver_world_model.training.objectives import make_jepa_objective
from torch.utils.data import DataLoader


def _batch(n=24, seed=0):
    sp = make_splits(n, 4, 4, horizon=36, history_length=12, prediction_horizon=12, seed=seed)
    dl = DataLoader(sp["train"], batch_size=16, shuffle=False, collate_fn=collate)
    return next(iter(dl))


def test_jepa_shapes():
    b = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12)
    out = m(b.history, b.hist_context, b.future, b.fut_context, b.current, b.ercp_future)
    assert out.pred_states.shape == b.future.shape
    assert out.pred_latents.shape == out.target_latents.shape
    assert out.context_latent.shape == (b.history.shape[0], 16)


def test_jepa_predictions_respect_constraints():
    """The core guarantee: predicted trajectories have zero hard-constraint violations."""
    b = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12)
    pred = m.predict(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future)
    v = constraint_violations(pred, b.ercp_future)
    assert int(v["total"]) == 0, {k: int(x) for k, x in v.items()}


def test_baseline_predictions_respect_constraints():
    b = _batch()
    m = DirectPredictor()
    pred = m(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future)
    assert int(constraint_violations(pred, b.ercp_future)["total"]) == 0


def test_backward_runs():
    b = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12)
    obj = make_jepa_objective()
    logs = obj(m, b, epoch=0)
    logs["loss"].backward()
    assert any(p.grad is not None for p in m.parameters())


def test_effective_dim_detects_collapse():
    d = 16
    spread = torch.randn(500, d)                      # full rank
    collapsed = torch.randn(500, 1) * torch.ones(1, d)  # rank-1
    assert compute_effective_dimensionality(spread) > 5
    assert compute_effective_dimensionality(collapsed) < 1.5


def test_predict_is_deterministic():
    b = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12).eval()
    p1 = m.predict(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future)
    p2 = m.predict(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future)
    assert torch.allclose(p1, p2)


def test_can_overfit_tiny_batch():
    """Sanity: the model can drive recon loss down on a single batch (learnable)."""
    b = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    obj = make_jepa_objective()
    first = None
    for _ in range(60):
        opt.zero_grad()
        logs = obj(m, b, epoch=20)
        logs["loss"].backward()
        opt.step()
        if first is None:
            first = float(logs["l_recon"])
    assert float(logs["l_recon"]) < first, "recon loss did not decrease"
