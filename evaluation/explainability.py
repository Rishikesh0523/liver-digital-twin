"""Explainability tooling.

A world model people can act on has to answer "why did you say that?". We give
three complementary views and then fuse them into one structured explanation for
a target month:

  * attention rollout  — which *fields* the encoder leaned on (causal-graph edges).
  * feature ablation    — zero each input field, measure prediction shift (a direct,
                          model-agnostic sensitivity that does not trust attention).
  * latent trajectory   — where h(t) travels; which latent dims move most.

We deliberately include ablation *alongside* attention because attention weights
are a seductive but unreliable explanation on their own — ablation is the check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.schema import FIELD_NAMES, STATE_DIM, cirrhosis_stage, F


def _batchify(history, hist_context, fut_context, current, ercp_future, device):
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    return (to(history), to(hist_context), to(fut_context), to(current), to(ercp_future))


class Explainability:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def attention_rollout(self, history, hist_context) -> np.ndarray:
        """Mean attention each field-node received, aggregated over heads/layers/time.

        Returns an (8,) vector: how much information flowed *into* each field node.
        Only meaningful for the JEPA (graph encoder); returns None otherwise.
        """
        enc = getattr(self.model, "encoder", None)
        if enc is None or not hasattr(enc, "mask"):
            return None
        h = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        c = torch.as_tensor(hist_context, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, attn_maps = enc(h, c, return_attn=True)
        # attn_maps: list of (B, T, heads, 8, 8). Average over everything but the
        # source axis (dim=-1) -> how much each field was attended TO.
        stacked = torch.stack([a.mean(dim=(0, 1, 2)) for a in attn_maps])  # (L, 8, 8)
        received = stacked.mean(0).mean(0)  # avg over layers and query nodes -> (8,)
        return received.cpu().numpy()

    @torch.no_grad()
    def feature_ablation(self, history, hist_context, fut_context, current,
                         ercp_future) -> dict[str, float]:
        """Zero each input field across history, measure change in predicted future.

        Returns per-field L2 shift in the predicted trajectory — a direct sensitivity.
        """
        args = _batchify(history, hist_context, fut_context, current, ercp_future, self.device)
        base = self._predict(*args)
        out = {}
        hist = args[0]
        for i in range(STATE_DIM):
            perturbed = hist.clone()
            perturbed[:, :, i] = 0.0
            pred = self._predict(perturbed, args[1], args[2], args[3], args[4])
            out[FIELD_NAMES[i]] = float((pred - base).pow(2).sum().sqrt())
        return out

    @torch.no_grad()
    def latent_trajectory(self, history, hist_context, fut_context) -> np.ndarray:
        """Predicted latent h_hat(t) over the horizon: (K, latent_dim)."""
        if not hasattr(self.model, "encoder"):
            return None
        h = torch.as_tensor(history, dtype=torch.float32, device=self.device).unsqueeze(0)
        c = torch.as_tensor(hist_context, dtype=torch.float32, device=self.device).unsqueeze(0)
        fc = torch.as_tensor(fut_context, dtype=torch.float32, device=self.device).unsqueeze(0)
        h0, _ = self.model.encoder(h, c)
        lat = self.model.predictor(h0, fc)  # (1, K, d)
        return lat.squeeze(0).cpu().numpy()

    def _predict(self, history, hist_context, fut_context, current, ercp_future):
        if hasattr(self.model, "predict"):  # JEPA
            return self.model.predict(history, hist_context, fut_context, current, ercp_future)
        return self.model(history, hist_context, fut_context, current, ercp_future)  # baseline

    # -- fused, human-readable explanation for a target month --------------
    def explain_month(self, history, hist_context, fut_context, current,
                      ercp_future, target_step: int) -> dict:
        """Structured 'why this prediction at target_step?' combining all three views."""
        args = _batchify(history, hist_context, fut_context, current, ercp_future, self.device)
        pred = self._predict(*args).squeeze(0).cpu().numpy()   # (K, 8)
        f_pred = pred[target_step, F]
        stage = int(cirrhosis_stage(f_pred))

        ablation = self.feature_ablation(history, hist_context, fut_context, current, ercp_future)
        attn = self.attention_rollout(history, hist_context)
        lat = self.latent_trajectory(history, hist_context, fut_context)

        drivers = sorted(ablation.items(), key=lambda kv: -kv[1])[:3]
        latent_move = None
        if lat is not None and target_step < len(lat):
            step_move = np.abs(lat[target_step] - lat[0])
            latent_move = {"top_dims": np.argsort(-step_move)[:3].tolist(),
                           "magnitudes": np.round(np.sort(step_move)[::-1][:3], 3).tolist()}

        return {
            "target_step": target_step,
            "predicted_state": {FIELD_NAMES[i]: round(float(pred[target_step, i]), 3)
                                for i in range(STATE_DIM)},
            "predicted_F": round(float(f_pred), 3),
            "derived_cirrhosis_stage": stage,
            "top_input_drivers_by_ablation": [(k, round(v, 3)) for k, v in drivers],
            "attention_received_by_field": (
                {FIELD_NAMES[i]: round(float(attn[i]), 3) for i in range(STATE_DIM)}
                if attn is not None else None),
            "latent_movement": latent_move,
        }
