"""Representation-collapse prevention and detection (Decisions D3, D8).

Collapse is the live failure mode of any learned latent: the encoder maps every
patient to (nearly) the same point, the predictor "predicts" it perfectly, the
latent-MSE loss goes to ~0, and the model has learned nothing. JEPA's stop-gradient
target removes the trivial way to reach collapse (predictor copying a live target),
but does not by itself *guarantee* a spread-out latent. So we add two cheap,
diagnostic regularisers and one metric that would catch collapse creeping back in.

Mechanism (prevention):
  * variance term  -> pushes each latent dimension to have std >= 1 (VICReg-style)
  * effective-rank term -> pushes the latent covariance toward full rank
Metric (detection):
  * participation ratio of the latent covariance eigenspectrum. Ranges in
    [1, latent_dim]; ~1 means fully collapsed, near latent_dim means well spread.
"""
from __future__ import annotations

import torch


def compute_effective_dimensionality(latents: torch.Tensor, eps: float = 1e-8) -> float:
    """Participation ratio of the covariance spectrum: (sum λ)^2 / sum(λ^2).

    latents: (N, d). Returns a float in [1, d]; near 1 == collapsed.
    """
    z = latents - latents.mean(0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = eig.sum()
    s2 = (eig * eig).sum()
    if float(s2) < eps:
        return 1.0
    return float((s1 * s1) / (s2 + eps))


class CollapseRegularizer(torch.nn.Module):
    """Variance + effective-rank regularisation on a batch of latents."""

    def __init__(self, gamma: float = 1.0, var_weight: float = 2.0,
                 rank_weight: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma          # target std per dimension
        self.var_weight = var_weight
        self.rank_weight = rank_weight
        self.eps = eps

    def forward(self, latents: torch.Tensor) -> dict[str, torch.Tensor]:
        # latents: (N, d)
        z = latents - latents.mean(0, keepdim=True)
        std = torch.sqrt(z.var(dim=0) + self.eps)           # (d,)
        # hinge: only penalise dimensions whose std fell below gamma
        var_loss = torch.relu(self.gamma - std).mean()

        # effective-rank surrogate: maximise normalised entropy of the eigenspectrum
        cov = (z.T @ z) / max(1, z.shape[0] - 1)
        eig = torch.linalg.eigvalsh(cov).clamp(min=self.eps)
        p = eig / eig.sum()
        entropy = -(p * torch.log(p)).sum()
        max_entropy = torch.log(torch.tensor(float(eig.numel())))
        rank_loss = 1.0 - entropy / max_entropy             # 0 = full rank, 1 = rank-1

        total = self.var_weight * var_loss + self.rank_weight * rank_loss
        with torch.no_grad():
            eff_dim = compute_effective_dimensionality(latents)
        return {
            "collapse_loss": total,
            "var_loss": var_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "effective_dim": torch.tensor(eff_dim),
        }
