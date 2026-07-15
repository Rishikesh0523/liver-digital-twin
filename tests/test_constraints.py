"""Phase-2 gate: the constrained parameterisation must produce valid states for
*arbitrary* network outputs — this is the "by construction" guarantee, so it must
hold for random garbage inputs, not just trained ones.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from liver_world_model.models.constraints import ConstrainedOutput, constraint_violations
from liver_world_model.data.schema import M, S, RATCHET_UP, upper_bounds


def _random_rollout(steps=20, batch=50, seed=0):
    """Roll the constrained layer forward with random deltas and random ERCP."""
    torch.manual_seed(seed)
    layer = ConstrainedOutput(dt=1.0)
    upper = torch.tensor(upper_bounds())
    # random valid start state in [0, upper]
    cur = torch.rand(batch, 8) * upper
    traj = [cur]
    ercp_log = []
    for t in range(steps):
        raw = torch.randn(batch, 8) * 5.0        # deliberately wild
        ercp = (torch.rand(batch) < 0.2).float()  # 20% ERCP months
        cur = layer(cur, raw, ercp)
        traj.append(cur)
        ercp_log.append(ercp)
    states = torch.stack(traj, dim=1)             # (B, steps+1, 8)
    # Destination-gated alignment: ercp_mask[k] is the flag used to PRODUCE
    # states[k]. states[0] has no incoming step, so prepend a zero.
    ercp_mask = torch.stack([torch.zeros(batch)] + ercp_log, dim=1)
    return states, ercp_mask


def test_no_violations_1000_states():
    """1000+ random states across random rollouts: zero hard-constraint violations."""
    states, ercp = _random_rollout(steps=25, batch=60, seed=1)  # 60*26 = 1560 states
    v = constraint_violations(states, ercp)
    assert int(v["total"]) == 0, f"violations: { {k: int(x) for k, x in v.items()} }"


def test_ratchet_holds_under_adversarial_input():
    """Even with raw deltas pushed strongly negative, ratchet fields never drop."""
    torch.manual_seed(2)
    layer = ConstrainedOutput()
    cur = torch.rand(100, 8) * torch.tensor(upper_bounds())
    raw = -50.0 * torch.ones(100, 8)             # try to force everything down
    nxt = layer(cur, raw, torch.zeros(100))
    for idx in RATCHET_UP:
        assert (nxt[:, idx] >= cur[:, idx] - 1e-6).all(), f"ratchet {idx} broke"


def test_M_zero_when_F_or_C_zero():
    """M increment is exactly zero when F or C is zero (F*C hazard coupling)."""
    from liver_world_model.data.schema import F, C
    layer = ConstrainedOutput()
    cur = torch.rand(50, 8) * torch.tensor(upper_bounds())
    cur[:, C] = 0.0                              # kill cholestasis
    raw = torch.randn(50, 8) * 3.0
    nxt = layer(cur, raw, torch.zeros(50))
    assert torch.allclose(nxt[:, M], cur[:, M], atol=1e-6), "M grew with C=0"


def test_S_can_only_drop_at_ercp():
    """S decreases only when the source month is flagged ERCP."""
    torch.manual_seed(3)
    layer = ConstrainedOutput()
    cur = torch.rand(200, 8) * torch.tensor(upper_bounds())
    cur[:, S] = 0.7                              # room to fall
    raw = torch.randn(200, 8)
    # off ERCP: never decreases
    nxt_off = layer(cur, raw, torch.zeros(200))
    assert (nxt_off[:, S] >= cur[:, S] - 1e-6).all(), "S dropped without ERCP"
    # at ERCP: allowed to fall (and with strongly negative raw, will)
    nxt_on = layer(cur, -5.0 * torch.ones(200, 8), torch.ones(200))
    assert (nxt_on[:, S] < cur[:, S]).any(), "ERCP never relieved S"


def test_bounds_respected():
    states, ercp = _random_rollout(steps=15, batch=40, seed=4)
    upper = torch.tensor(upper_bounds())
    assert (states >= -1e-6).all()
    assert (states <= upper + 1e-6).all()
