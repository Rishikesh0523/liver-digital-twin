"""Composite JEPA loss.

    L = w_pred*L_pred + w_constr*L_constr + w_collapse*L_collapse + w_recon*L_recon

* L_pred    : latent-space MSE between predicted and stop-grad target latents.
              This is the JEPA core — the model is graded on predicting the
              *representation* of the future, not its raw values.
* L_constr  : soft penalty on any hard-constraint violation. With the constrained
              parameterisation this should be identically 0; we keep it as a live
              tripwire (Decision D7) — a nonzero value means the guarantee broke.
* L_collapse: variance + effective-rank regularisation on the latents (memo §2).
* L_recon   : state-space MSE, keeping the decoder honest and the latent decodable
              to an auditable clinical state.

Weight schedule (memo §3): constraints are weighted heavily early so the decoder
learns the increment regime fast, then relaxed so prediction dominates.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..models.collapse import CollapseRegularizer
from ..models.constraints import constraint_violations


@dataclass
class LossWeights:
    w_pred: float = 1.0
    w_constr: float = 10.0
    w_collapse: float = 0.1
    w_recon: float = 0.5


def schedule_weights(epoch: int, base: LossWeights, decay_epochs: int = 10) -> LossWeights:
    """Relax the constraint weight (10->1) after warmup and *raise* the collapse
    weight (0.1->1.0). Rationale (memo §2): the constrained decoder conditions on
    the current state and can reconstruct the future from it alone, giving the
    latent a shortcut to collapse. Once constraints are learned, collapse becomes
    the dominant risk, so its weight has to grow, not shrink."""
    late = LossWeights(w_pred=1.0, w_constr=1.0, w_collapse=1.0, w_recon=0.25)
    if epoch >= decay_epochs:
        return late
    frac = epoch / max(1, decay_epochs)
    return LossWeights(
        w_pred=base.w_pred,
        w_constr=base.w_constr + (late.w_constr - base.w_constr) * frac,
        w_collapse=base.w_collapse + (late.w_collapse - base.w_collapse) * frac,
        w_recon=base.w_recon + (late.w_recon - base.w_recon) * frac,
    )


class JEPALoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None, w_anchor: float = 0.5):
        super().__init__()
        self.collapse = CollapseRegularizer()
        self.weights = weights or LossWeights()
        self.w_anchor = w_anchor  # denoised-anchor reconstruction weight

    def forward(self, out, batch, weights: LossWeights | None = None) -> dict:
        w = weights or self.weights
        # --- latent prediction (JEPA core) --------------------------------
        l_pred = F_.mse_loss(out.pred_latents, out.target_latents)
        # --- state reconstruction -----------------------------------------
        l_recon = F_.mse_loss(out.pred_states, batch.future)
        # --- collapse regularisation --------------------------------------
        # Regularise BOTH the predictor output AND the encoder's context latent.
        # The encoder is the *root* of collapse (targets are detached, so only the
        # context latent carries gradient back into the encoder); pushing distinct
        # patients to distinct h is what actually prevents the encoder degenerating.
        flat = out.pred_latents.reshape(-1, out.pred_latents.shape[-1])
        coll = self.collapse(flat)
        coll_ctx = self.collapse(out.context_latent)
        l_collapse = coll["collapse_loss"] + coll_ctx["collapse_loss"]
        # --- constraint tripwire (should be 0 by construction) ------------
        with torch.no_grad():
            v = constraint_violations(out.pred_states, batch.ercp_future)
            viol = v["total"].float()
        # differentiable backup penalty (stays 0 under the parameterisation)
        l_constr = _soft_constraint_penalty(out.pred_states, batch.current, batch.ercp_future)
        # --- denoised-anchor reconstruction: teach the latent to recover the true
        # current state from history (trained on clean data; pays off under noise) ---
        l_anchor = F_.mse_loss(out.anchor_estimate, batch.current)

        total = (w.w_pred * l_pred + w.w_constr * l_constr
                 + w.w_collapse * l_collapse + w.w_recon * l_recon
                 + self.w_anchor * l_anchor)
        return {
            "loss": total,
            # Model-SELECTION metric = state-reconstruction error (what the eval
            # actually measures). NOT the total loss (its collapse penalty grows over
            # training) and NOT latent-pred (a *collapsed* latent has the LOWEST
            # latent-pred — collapse is an attractor because it predicts itself
            # perfectly). Selection is collapse-GUARDED in the trainer: minimise this
            # subject to effective_dim clearing the health bar.
            "monitor": l_recon.detach(),
            "l_pred": l_pred.detach(),
            "l_recon": l_recon.detach(),
            "l_collapse": l_collapse.detach(),
            "l_constr": l_constr.detach(),
            "l_anchor": l_anchor.detach(),
            "violations": viol,
            # report the *context* latent's effective dim — the quantity the eval
            # harness measures on the encoder output (the root collapse signal).
            "effective_dim": coll_ctx["effective_dim"],
            "effective_dim_pred": coll["effective_dim"],
            "var_loss": coll["var_loss"],
            "rank_loss": coll["rank_loss"],
        }


def _soft_constraint_penalty(pred_states: torch.Tensor, current: torch.Tensor,
                             ercp_future: torch.Tensor) -> torch.Tensor:
    """Differentiable backup penalty. Zero for a correct parameterisation, but if a
    future refactor ever breaks the by-construction guarantee this term still
    pushes back. Penalises: ratchet decreases and out-of-bounds mass."""
    from ..data.schema import RATCHET_UP, upper_bounds
    upper = torch.tensor(upper_bounds(), device=pred_states.device)
    # prepend current to compute step diffs across the whole predicted window
    seq = torch.cat([current.unsqueeze(1), pred_states], dim=1)     # (B, K+1, 8)
    diffs = seq[:, 1:] - seq[:, :-1]
    pen = pred_states.new_zeros(())
    for idx in RATCHET_UP:
        pen = pen + torch.relu(-diffs[:, :, idx]).mean()
    pen = pen + torch.relu(-pred_states).mean()
    pen = pen + torch.relu(pred_states - upper).mean()
    return pen
