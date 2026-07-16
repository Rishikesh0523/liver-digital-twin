"""Loss closures binding each model to the generic Trainer.

Both return a dict with a ``loss`` key plus scalar diagnostics, so the trainer's
logging/early-stopping treat them identically.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F_

from ..models.constraints import constraint_violations
from .losses import JEPALoss, LossWeights, schedule_weights


def make_jepa_objective(base_weights: LossWeights | None = None):
    lossfn = JEPALoss()
    base = base_weights or LossWeights()

    def objective(model, batch, epoch: int = 0):
        out = model(batch.history, batch.hist_context, batch.future,
                    batch.fut_context, batch.current, batch.ercp_future)
        w = schedule_weights(epoch, base)
        return lossfn(out, batch, weights=w)

    return objective


def make_baseline_objective():
    """Baseline is graded on state-space MSE (it has no latent to predict)."""

    def objective(model, batch, epoch: int = 0):
        pred = model(batch.history, batch.hist_context, batch.fut_context,
                     batch.current, batch.ercp_future)
        l_recon = F_.mse_loss(pred, batch.future)
        with torch.no_grad():
            viol = constraint_violations(pred, batch.ercp_future)["total"].float()
        return {
            "loss": l_recon,
            "l_recon": l_recon.detach(),
            "l_pred": torch.tensor(0.0),
            "l_collapse": torch.tensor(0.0),
            "l_constr": torch.tensor(0.0),
            "violations": viol,
            "effective_dim": torch.tensor(float("nan")),
        }

    return objective
