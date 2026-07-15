"""Constraint enforcement by construction (Decision D1).

This is the heart of the "auditability vs. expressiveness" tradeoff. We choose a
**constrained parameterisation** over post-hoc projection: the decoder never emits
an absolute state, it emits *deltas relative to the current state*, and those
deltas are passed through sign-fixing nonlinearities. The one-directional fields
therefore cannot reverse — not because the loss learned it, but because the
functional form makes reversal unrepresentable.

Guarantees (proved by the shapes of the transforms, tested in test_constraints.py):
  * F, D, P  : x' = x + softplus(.)            => x' >= x           (ratchet up)
  * M        : x' = x + softplus(.)*F*C*dt      => x' >= x, and the
               increment is *structurally* the F*C hazard coupling (Decision D4)
  * S        : x' = x + softplus(.)  off ERCP   => x' >= x
               x' = sigmoid(.)       at ERCP    => free in [0,1]   (Decision D2)
  * A, C, fl : x' = sigmoid(.)                   => in [0,1]        (free/fast)
  * bounds   : clamp to [0, upper] as a numerical backstop.

Cost (paid honestly in the memo §3): the latent cannot express a "shortcut"
that would momentarily dip a ratchet field; the optimisation landscape is
harder; and the M coupling is *imposed*, so if the true hazard were not F*C the
model could not discover that. We accept this: clinical safety > flexibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..data.schema import (
    A, C, D, F, FLARE, M, P, S, STATE_DIM, RATCHET_UP, FREE, upper_bounds,
)


class ConstrainedOutput(nn.Module):
    """Map (current_state, raw_deltas, ercp_flag) -> valid next_state.

    Args:
        dt: month step used for the M hazard accumulation.
    """

    def __init__(self, dt: float = 1.0, delta_shift: float = 4.0):
        super().__init__()
        self.dt = dt
        # Init prior: ratchet deltas are softplus(raw - delta_shift). At init raw~=0,
        # so softplus(-4)~=0.018 per step instead of softplus(0)~=0.69 — without this
        # an untrained decoder saturates every ratchet field to its ceiling in ~2
        # steps and has to *learn* strongly-negative outputs just to stand still.
        self.delta_shift = delta_shift
        upper = torch.tensor(upper_bounds(), dtype=torch.float32)
        self.register_buffer("upper", upper)  # (8,)
        ratchet_mask = torch.zeros(STATE_DIM)
        for idx in RATCHET_UP:
            ratchet_mask[idx] = 1.0
        self.register_buffer("ratchet_mask", ratchet_mask)  # (8,)

    def forward(self, current: torch.Tensor, raw: torch.Tensor,
                ercp: torch.Tensor) -> torch.Tensor:
        """current: (B, 8) previous state; raw: (B, 8) unconstrained network output;
        ercp: (B,) or (B,1) in {0,1}. Returns next state (B, 8), guaranteed valid."""
        if ercp.dim() == 1:
            ercp = ercp.unsqueeze(-1)
        ercp_s = ercp[:, 0]
        # Build each column functionally (no in-place writes) so autograd is happy.
        cols = [None] * STATE_DIM

        sh = self.delta_shift
        # --- ratchet-up fields F, D, P: non-negative delta -----------------
        for idx in (F, D, P):
            cols[idx] = current[:, idx] + F_.softplus(raw[:, idx] - sh)

        # --- M: hazard of sustained F*C, embedded in the parameterisation ---
        # dM = softplus(raw_M) * F * C * dt.  Uses the *current* F,C as the
        # generator does; >=0 so M ratchets; ==0 exactly when F or C is 0.
        dM = F_.softplus(raw[:, M] - sh) * current[:, F] * current[:, C] * self.dt
        cols[M] = current[:, M] + dM

        # --- S: ratchet up off-ERCP, free (step-down allowed) at ERCP -------
        s_up = current[:, S] + F_.softplus(raw[:, S] - sh)     # non-decreasing branch
        s_free = torch.sigmoid(raw[:, S]) * self.upper[S]       # free branch (relief)
        cols[S] = ercp_s * s_free + (1.0 - ercp_s) * s_up

        # --- free/fast fields A, C, flare: sigmoid into [0,1] --------------
        for idx in (A, C, FLARE):
            cols[idx] = torch.sigmoid(raw[:, idx]) * self.upper[idx]

        out = torch.stack(cols, dim=-1)                        # (B, 8)
        # numerical backstop: clamp to [0, upper], then re-assert the ratchet floor
        out = torch.clamp(out, min=torch.zeros_like(self.upper), max=self.upper)
        ratchet_floor = current * self.ratchet_mask            # 0 for free fields
        out = torch.maximum(out, ratchet_floor)
        return out


@torch.no_grad()
def constraint_violations(states: torch.Tensor, ercp: torch.Tensor,
                          tol: float = 1e-5) -> dict[str, torch.Tensor]:
    """Count hard-constraint violations over a batch of trajectories.

    states: (B, T, 8); ercp: (B, T) in {0,1}. Returns per-constraint violation
    counts (as tensors) — used both in tests and as an honest eval metric.
    A correct by-construction model returns all zeros (Decision D7).
    """
    upper = torch.tensor(upper_bounds(), device=states.device)
    B, T, _ = states.shape
    diffs = states[:, 1:] - states[:, :-1]            # (B, T-1, 8)
    out: dict[str, torch.Tensor] = {}

    # ratchet-up fields must never decrease
    for idx, name in ((F, "F"), (D, "D"), (P, "P"), (M, "M")):
        out[f"mono_{name}"] = (diffs[:, :, idx] < -tol).sum()

    # S may only decrease *into* an ERCP month (destination-gated, D2):
    # the drop from t -> t+1 is legal iff month t+1 is an ERCP month.
    ds = diffs[:, :, S]                                # step t -> t+1
    ercp_dst = ercp[:, 1:] > 0.5                       # ERCP at destination month
    illegal_s = (ds < -tol) & (~ercp_dst)
    out["mono_S_nonercp"] = illegal_s.sum()

    # bounds
    below = (states < -tol).sum()
    above = (states > upper + tol).sum()
    out["bound_low"] = below
    out["bound_high"] = above
    out["total"] = sum(v for v in out.values())
    return out
