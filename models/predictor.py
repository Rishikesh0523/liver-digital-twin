"""Latent dynamics predictor.

Rolls the encoder latent ``h`` forward in latent space, conditioned on the
*time-varying* context at each future step (so UDCA onset / ERCP events enter the
rollout). We use a GRU cell, not a Neural-ODE, and say why in the docstring —
this is a deliberate call, not an omission.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.schema import CONTEXT_DIM


class LatentPredictor(nn.Module):
    """Discrete-time latent rollout: h_{t+1} = GRUCell(context_t, h_t).

    Why GRU over Neural-ODE here (Decision, memo §1):
        The generator emits *fixed monthly* steps — there is no irregular sampling
        for a continuous-time model to exploit, and Euler-integrating an ODE at
        dt=1 month collapses to a residual GRU anyway but with more machinery to
        babysit (stiffness, solver tolerance). A GRU gives the same expressivity
        for this grid at a fraction of the training cost/instability. The ODE is
        the right extension the moment sampling becomes irregular (labs at
        arbitrary times), and the interface here (step-by-step) drops in cleanly.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 64,
                 ctx_dim: int = CONTEXT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.cell = nn.GRUCell(hidden, latent_dim)

    def step(self, h: torch.Tensor, context_t: torch.Tensor) -> torch.Tensor:
        return self.cell(self.ctx_encoder(context_t), h)

    def forward(self, h0: torch.Tensor, future_context: torch.Tensor) -> torch.Tensor:
        """h0: (B, latent_dim); future_context: (B, K, ctx_dim).
        Returns predicted latents (B, K, latent_dim), one per future step."""
        B, K, _ = future_context.shape
        h = h0
        outs = []
        for t in range(K):
            h = self.step(h, future_context[:, t])
            outs.append(h)
        return torch.stack(outs, dim=1)
