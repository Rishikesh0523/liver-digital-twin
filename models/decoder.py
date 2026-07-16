"""Constrained decoder: predicted latent -> valid next state.

The decoder is where "latent space is free to violate constraints" gets fixed.
It emits *raw deltas*, and hands them (with the current state and the ERCP flag)
to :class:`ConstrainedOutput`, which makes the one-directional guarantees hold by
construction. Because the ratchet base is the current state, the decoder only ever
has to learn *increments*, which also keeps the slow fields numerically stable
over long rollouts.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.schema import STATE_DIM
from .constraints import ConstrainedOutput


class ConstrainedDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden=(64, 32), dt: float = 1.0):
        super().__init__()
        dims = [latent_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU()]
        layers += [nn.Linear(dims[-1], STATE_DIM)]
        self.net = nn.Sequential(*layers)
        self.constrain = ConstrainedOutput(dt=dt)

    def forward(self, latent: torch.Tensor, current: torch.Tensor,
                ercp: torch.Tensor) -> torch.Tensor:
        """latent: (B, latent_dim); current: (B, 8); ercp: (B,). -> next state (B,8)."""
        raw = self.net(latent)
        return self.constrain(current, raw, ercp)

    def rollout(self, latents: torch.Tensor, current: torch.Tensor,
                ercp_future: torch.Tensor) -> torch.Tensor:
        """Autoregressive decode over K steps, threading the ratchet base forward.

        latents: (B, K, latent_dim); current: (B, 8) last observed state;
        ercp_future: (B, K). Returns predicted states (B, K, 8)."""
        B, K, _ = latents.shape
        cur = current
        outs = []
        for t in range(K):
            cur = self.forward(latents[:, t], cur, ercp_future[:, t])
            outs.append(cur)
        return torch.stack(outs, dim=1)
