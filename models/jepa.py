"""Full JEPA-style world model.

Composition:
    encoder   : history (+context)          -> h                (context latent)
    predictor : h, future_context           -> h_hat[1..K]      (latent rollout)
    decoder   : h_hat[t], current, ercp_t   -> x_hat[t]         (valid state)

Training targets live in **latent space** (JEPA proper): the target latent for
future step t is a stop-gradient encoding of the true trajectory *up to* that
step. Predicting the representation rather than raw values is what lets the latent
carry unobserved structure (per-patient "drive") that the 8-D state never exposes —
the reason we go JEPA over predicting x(t) directly (memo §1).

A state-space reconstruction head is kept (w_recon) so the decoder stays honest
and the latent stays *decodable* to an auditable clinical state — without it, the
latent could drift into a private code that predicts itself but means nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..data.schema import STATE_DIM, CONTEXT_DIM
from .encoder import GraphAttentionEncoder
from .predictor import LatentPredictor
from .decoder import ConstrainedDecoder


@dataclass
class JEPAOutput:
    pred_latents: torch.Tensor      # (B, K, latent_dim) predicted future latents
    target_latents: torch.Tensor    # (B, K, latent_dim) stop-grad targets
    pred_states: torch.Tensor       # (B, K, 8) decoded valid states
    context_latent: torch.Tensor    # (B, latent_dim) encoder summary h
    anchor_estimate: torch.Tensor   # (B, 8) history-denoised estimate of current state


class LiverJEPA(nn.Module):
    def __init__(self, latent_dim: int = 16, d_model: int = 32, n_heads: int = 4,
                 n_attn_layers: int = 2, predictor_hidden: int = 64,
                 decoder_hidden=(64, 32), history_length: int = 12, dt: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.encoder = GraphAttentionEncoder(
            latent_dim=latent_dim, d_model=d_model, n_heads=n_heads,
            n_layers=n_attn_layers,
        )
        self.predictor = LatentPredictor(latent_dim=latent_dim, hidden=predictor_hidden)
        self.decoder = ConstrainedDecoder(latent_dim=latent_dim, hidden=decoder_hidden, dt=dt)
        # Denoised anchor: a small head that reconstructs the *current* clinical state
        # from the history latent. Because the latent integrates the whole 12-month
        # window, this estimate averages out per-visit observation noise — so at
        # inference under noisy observations we can anchor the ratchet rollout on this
        # instead of the single (noisy) last observation. A memoryless/direct model
        # has no equivalent. This is where the predictive latent earns its keep
        # (measured in the noise-robustness probe; see memo §Results-B).
        self.anchor_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, STATE_DIM)
        )
        self.register_buffer("_upper", torch.tensor(
            [1., 1., 1., 1., 1., 1., 2., 1.], dtype=torch.float32))

    def _anchor_from_latent(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.anchor_head(h)) * self._upper

    # -- target latents: stop-grad encodings of the true future ------------
    @torch.no_grad()
    def encode_targets(self, history: torch.Tensor, hist_context: torch.Tensor,
                       future: torch.Tensor, fut_context: torch.Tensor) -> torch.Tensor:
        """For each future step t, encode the window ending at t (teacher-forced,
        stop-gradient) as the prediction target. Sliding window keeps the target a
        *representation of the real state up to t*, not of the raw value alone.
        """
        B, K, _ = future.shape
        H = history.shape[1]
        full = torch.cat([history, future], dim=1)              # (B, H+K, 8)
        full_ctx = torch.cat([hist_context, fut_context], dim=1)
        # ONE encoder pass over [history+future]; take the per-timestep readout at
        # the future positions H..H+K-1 as the K targets. Target for future step t is
        # the representation of the trajectory *up to* that state (stop-gradient).
        seq, _ = self.encoder(full, full_ctx, return_sequence=True)  # (B, H+K, latent)
        return seq[:, H:H + K].detach()                         # (B, K, latent_dim)

    def forward(self, history: torch.Tensor, hist_context: torch.Tensor,
                future: torch.Tensor, fut_context: torch.Tensor,
                current: torch.Tensor, ercp_future: torch.Tensor,
                with_targets: bool = True) -> JEPAOutput:
        h, _ = self.encoder(history, hist_context)              # (B, latent_dim)
        pred_latents = self.predictor(h, fut_context)           # (B, K, latent_dim)
        pred_states = self.decoder.rollout(pred_latents, current, ercp_future)
        anchor = self._anchor_from_latent(h)                    # (B, 8) denoised current
        targets = (self.encode_targets(history, hist_context, future, fut_context)
                   if with_targets else torch.zeros_like(pred_latents))
        return JEPAOutput(pred_latents, targets, pred_states, h, anchor)

    # -- inference-time rollout (no ground-truth future needed) ------------
    @torch.no_grad()
    def predict(self, history: torch.Tensor, hist_context: torch.Tensor,
                fut_context: torch.Tensor, current: torch.Tensor,
                ercp_future: torch.Tensor, denoise: bool = False) -> torch.Tensor:
        h, _ = self.encoder(history, hist_context)
        pred_latents = self.predictor(h, fut_context)
        # denoise=True anchors the ratchet rollout on the history-decoded state
        # estimate instead of the (possibly noisy) observed `current`.
        base = self._anchor_from_latent(h) if denoise else current
        return self.decoder.rollout(pred_latents, base, ercp_future)
