"""Learned manifold critic — "0 violations != on-manifold".

Constraint satisfaction is necessary but not sufficient. A trajectory can honour
every monotonicity/bound and still be *dynamically wrong* (right direction, wrong
magnitude), i.e. off the manifold of transitions the generator actually produces.

We train a small discriminator to separate **real** generator transitions
`(x_t -> x_{t+1})` from **constraint-valid-but-wrong** ones, then score each model's
predicted transitions with it. A model whose predictions the critic can't
distinguish from real (score ~ real-data score) is on-manifold; a model the critic
easily flags is drifting even at 0% violations.

Negatives are generated to be *constraint-valid* (so the critic must learn dynamics,
not just legality): we take a real `x_t` and apply a valid-but-mis-scaled increment
(random monotone deltas within bounds, ERCP-consistent) rather than the true one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..data.schema import STATE_DIM, RATCHET_UP, S as S_IDX, upper_bounds
from ..models.constraints import ConstrainedOutput


class _Critic(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        # input = (x_t, x_{t+1}, ercp_flag) -> real/fake logit
        self.net = nn.Sequential(
            nn.Linear(2 * STATE_DIM + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_t, x_next, ercp):
        z = torch.cat([x_t, x_next, ercp.unsqueeze(-1)], dim=-1)
        return self.net(z).squeeze(-1)


@dataclass
class ManifoldReport:
    critic_auc: float                 # how well the critic separates real vs valid-wrong
    score_real: float                 # mean critic "realness" prob on held-out real transitions
    scores: dict                      # model_name -> mean realness prob on its transitions
    note: str


def _valid_wrong_negatives(x_t: torch.Tensor, ercp: torch.Tensor,
                           rng: torch.Generator) -> torch.Tensor:
    """Constraint-valid but dynamically wrong next states, via the same
    ConstrainedOutput head fed random raw deltas (so legality holds, dynamics don't)."""
    head = ConstrainedOutput()
    raw = torch.randn(x_t.shape, generator=rng) * 2.0
    return head(x_t, raw, ercp)


def _transitions_from_states(states: np.ndarray, ercp: np.ndarray):
    """states: (N, T, 8), ercp: (N, T). Return (x_t, x_next, ercp_dest) flattened."""
    x_t = torch.tensor(states[:, :-1].reshape(-1, STATE_DIM), dtype=torch.float32)
    x_next = torch.tensor(states[:, 1:].reshape(-1, STATE_DIM), dtype=torch.float32)
    ercp_dest = torch.tensor(ercp[:, 1:].reshape(-1), dtype=torch.float32)
    return x_t, x_next, ercp_dest


def train_critic(real_states: np.ndarray, real_ercp: np.ndarray, device,
                 epochs: int = 60, seed: int = 0) -> tuple[_Critic, float, float]:
    """Train the critic on real vs valid-wrong transitions; return (critic, auc, real_score)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    x_t, x_next, ercp = _transitions_from_states(real_states, real_ercp)
    n = x_t.shape[0]
    split = int(0.8 * n)
    critic = _Critic().to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(split, generator=gen)
        xt, xn, ec = x_t[perm], x_next[perm], ercp[perm]
        neg = _valid_wrong_negatives(xt, ec, gen)
        opt.zero_grad()
        real_logit = critic(xt.to(device), xn.to(device), ec.to(device))
        fake_logit = critic(xt.to(device), neg.to(device), ec.to(device))
        loss = (F_.binary_cross_entropy_with_logits(real_logit, torch.ones_like(real_logit))
                + F_.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit)))
        loss.backward()
        opt.step()
    # eval AUC on held-out real vs fresh valid-wrong
    critic.eval()
    with torch.no_grad():
        xt, xn, ec = x_t[split:], x_next[split:], ercp[split:]
        neg = _valid_wrong_negatives(xt, ec, gen)
        rp = torch.sigmoid(critic(xt.to(device), xn.to(device), ec.to(device))).cpu().numpy()
        fp = torch.sigmoid(critic(xt.to(device), neg.to(device), ec.to(device))).cpu().numpy()
    auc = _auc(rp, fp)
    return critic, auc, float(rp.mean())


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC: P(score(pos) > score(neg))."""
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = len(pos), len(neg)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def score_model_transitions(critic: _Critic, pred_states: torch.Tensor,
                            current: torch.Tensor, ercp_future: torch.Tensor,
                            device) -> float:
    """Mean critic realness prob over a model's predicted transitions.

    pred_states: (B, K, 8); current: (B, 8); ercp_future: (B, K).
    We prepend `current` so the first predicted step is a transition too."""
    seq = torch.cat([current.unsqueeze(1), pred_states], dim=1)   # (B, K+1, 8)
    x_t = seq[:, :-1].reshape(-1, STATE_DIM)
    x_next = seq[:, 1:].reshape(-1, STATE_DIM)
    ercp = ercp_future.reshape(-1)
    probs = torch.sigmoid(critic(x_t.to(device), x_next.to(device), ercp.to(device)))
    return float(probs.mean())
