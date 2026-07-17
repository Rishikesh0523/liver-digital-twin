"""Fast sanity tests for the extended evaluation (denoised anchor, manifold critic,
counterfactual). These use tiny untrained models — they check the *mechanics* hold
(shapes, constraint preservation, discriminability), not trained-model quality.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from liver_world_model.data.generator import LiverDiseaseGenerator
from liver_world_model.data.dataset import make_splits, collate
from liver_world_model.models.jepa import LiverJEPA
from liver_world_model.models.constraints import constraint_violations
from liver_world_model.evaluation import manifold_critic
from torch.utils.data import DataLoader


def _batch(seed=0):
    sp = make_splits(20, 4, 4, horizon=36, history_length=12, prediction_horizon=12, seed=seed)
    return next(iter(DataLoader(sp["train"], batch_size=16, collate_fn=collate))), sp


def test_denoised_anchor_shapes_and_constraints():
    """Denoised-anchor prediction has the right shape and still 0 violations."""
    b, _ = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12).eval()
    p_raw = m.predict(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future,
                      denoise=False)
    p_den = m.predict(b.history, b.hist_context, b.fut_context, b.current, b.ercp_future,
                      denoise=True)
    assert p_den.shape == p_raw.shape == b.future.shape
    assert not torch.allclose(p_raw, p_den), "denoise flag had no effect"
    assert int(constraint_violations(p_den, b.ercp_future)["total"]) == 0


def test_anchor_estimate_in_bounds():
    """The anchor state estimate is a valid state (in [0, upper])."""
    from liver_world_model.data.schema import upper_bounds
    b, _ = _batch()
    m = LiverJEPA(latent_dim=16, history_length=12)
    out = m(b.history, b.hist_context, b.future, b.fut_context, b.current, b.ercp_future)
    upper = torch.tensor(upper_bounds())
    assert (out.anchor_estimate >= -1e-6).all()
    assert (out.anchor_estimate <= upper + 1e-6).all()


def test_manifold_critic_separates_real_from_valid_wrong():
    """The critic learns to distinguish real transitions from constraint-valid-but-wrong
    ones — i.e. constraint satisfaction alone is not enough to look real (AUC > 0.8)."""
    gen = LiverDiseaseGenerator()
    trajs = gen.generate(120, horizon=30, seed=1)
    states = np.stack([t.states for t in trajs])
    ercp = np.stack([t.ercp_mask for t in trajs]).astype(float)
    _, auc, real_score = manifold_critic.train_critic(states, ercp, torch.device("cpu"),
                                                      epochs=40, seed=1)
    assert auc > 0.8, f"critic failed to separate (AUC={auc})"
    assert real_score > 0.5, "real transitions should score above chance"


def test_counterfactual_runs_and_reports():
    """Counterfactual probe runs end-to-end on an untrained model and returns a report
    with a defined sign-agreement (mechanics, not quality)."""
    from liver_world_model.evaluation import counterfactual
    m = LiverJEPA(latent_dim=16, history_length=12).eval()
    rep = counterfactual.run(m, torch.device("cpu"), history_length=12,
                             prediction_horizon=12, n_patients=20)
    assert rep.outcome_field == "C"
    assert 0.0 <= rep.sign_agreement_rate <= 1.0 or rep.sign_agreement_rate != rep.sign_agreement_rate
    assert isinstance(rep.verdict, str) and len(rep.verdict) > 0
