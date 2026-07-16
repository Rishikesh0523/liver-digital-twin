"""Baseline: direct autoregressive next-state predictor.

The "simple peer" the brief asks us to weigh JEPA against: x(t) *is* the latent,
no representation learning. A GRU reads the history, then predicts the next state
one step at a time. It uses the **same** constrained output head as JEPA, so the
comparison is clean — the only thing that differs is JEPA's predictive latent, not
the constraint mechanism. Any accuracy gap is therefore attributable to the latent,
and any *constraint* parity confirms the guarantee is architectural, not model-specific.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.schema import STATE_DIM, CONTEXT_DIM
from .constraints import ConstrainedOutput


class DirectPredictor(nn.Module):
    def __init__(self, hidden: int = 64, ctx_dim: int = CONTEXT_DIM, dt: float = 1.0):
        super().__init__()
        self.encoder = nn.GRU(STATE_DIM + ctx_dim, hidden, batch_first=True)
        self.step_cell = nn.GRUCell(STATE_DIM + ctx_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, STATE_DIM)
        )
        self.constrain = ConstrainedOutput(dt=dt)

    def forward(self, history: torch.Tensor, hist_context: torch.Tensor,
                fut_context: torch.Tensor, current: torch.Tensor,
                ercp_future: torch.Tensor) -> torch.Tensor:
        """Predict K future states autoregressively. Returns (B, K, 8)."""
        h_in = torch.cat([history, hist_context], dim=-1)
        _, hidden = self.encoder(h_in)                 # (1, B, hidden)
        hidden = hidden.squeeze(0)
        K = fut_context.shape[1]
        cur = current
        outs = []
        for t in range(K):
            inp = torch.cat([cur, fut_context[:, t]], dim=-1)
            hidden = self.step_cell(inp, hidden)
            raw = self.head(hidden)
            cur = self.constrain(cur, raw, ercp_future[:, t])
            outs.append(cur)
        return torch.stack(outs, dim=1)
