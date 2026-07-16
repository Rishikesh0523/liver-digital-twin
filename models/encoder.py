"""Graph-attention encoder (Decision D3).

Encodes a patient's history into a latent ``h in R^latent_dim``. Two structural
priors from the brief are baked in:

1. **Node-per-field.** Each of the 8 state fields is a node; attention runs over
   the 8 nodes so the encoder reasons about field *interactions*, not a flat vector.

2. **Causal attention mask.** Node i may attend to node j only if j biologically
   influences i (the generator's dependency graph). This is the "information flows
   along biological edges" idea — and gives the referral-bias probe something to
   bite on: a masked model cannot fabricate a shortcut edge that isn't in the graph.

Temporal structure is then folded in with a GRU over the per-timestep graph
summaries. The final latent is a linear readout — small enough (16) that collapse
would be visible, large enough to hold an unobserved "drive" the 8-D state omits.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_

from ..data.schema import (
    A, C, D, F, FLARE, M, P, S, STATE_DIM, CONTEXT_DIM,
)


def build_causal_mask() -> torch.Tensor:
    """Boolean (8,8) mask; mask[i,j]=True means node i may attend to node j.

    Edges are 'target attends to its biological sources' (+ self-loops), read
    straight off the generator dynamics so the prior matches the true process.
    """
    m = torch.eye(STATE_DIM, dtype=torch.bool)
    edges = {
        F: (A, C),          # fibrosis driven by inflammation + cholestasis
        D: (C,),            # ductopenia driven by cholestasis
        S: (A,),            # strictures driven by inflammatory activity
        P: (F,),            # portal hypertension driven by fibrosis
        A: (C, FLARE),      # activity <-> cholestasis, spikes on flare
        C: (A, FLARE),      # cholestasis <-> activity, spikes on flare (treatment via context)
        M: (F, C),          # malignancy hazard from sustained F*C
        FLARE: (A, C),      # flare triggered by activity/cholestasis
    }
    for tgt, srcs in edges.items():
        for s in srcs:
            m[tgt, s] = True
    return m


class GraphAttentionLayer(nn.Module):
    """Single masked multi-head self-attention over the 8 field-nodes."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                return_attn: bool = False):
        # x: (N, 8, d_model); mask: (8, 8) bool
        N, T, dm = x.shape
        qkv = self.qkv(self.norm(x)).reshape(N, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (N, heads, 8, d_head)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (N, heads, 8, 8)
        scores = scores.masked_fill(~mask.view(1, 1, T, T), float("-inf"))
        attn = F_.softmax(scores, dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(N, T, dm)
        x = x + self.proj(ctx)
        x = x + self.ff(self.norm2(x))
        if return_attn:
            return x, attn
        return x, None


class GraphAttentionEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        # per-node scalar -> d_model (each field gets its own projection weights)
        self.value_proj = nn.Parameter(torch.randn(STATE_DIM, d_model) * 0.1)
        self.value_bias = nn.Parameter(torch.zeros(STATE_DIM, d_model))
        self.node_emb = nn.Parameter(torch.randn(STATE_DIM, d_model) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.ctx_proj = nn.Linear(CONTEXT_DIM, d_model)
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.register_buffer("mask", build_causal_mask())
        # temporal aggregation over per-timestep node summaries
        self.node_pool = nn.Linear(STATE_DIM * d_model, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.readout = nn.Linear(d_model, latent_dim)

    def _embed_nodes(self, states: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # states: (B, T, 8); context: (B, T, CONTEXT_DIM)
        B, T, _ = states.shape
        # scalar value -> per-node feature
        nodes = states.unsqueeze(-1) * self.value_proj + self.value_bias  # (B,T,8,d)
        nodes = nodes + self.node_emb.view(1, 1, STATE_DIM, self.d_model)
        nodes = nodes + self.time_emb[:T].view(1, T, 1, self.d_model)
        # context bias broadcast to every node (treatment/ERCP signal is global,
        # its targeted effect is learned through which nodes can propagate it)
        ctx = self.ctx_proj(context).view(B, T, 1, self.d_model)
        nodes = nodes + ctx
        return nodes  # (B, T, 8, d)

    def forward(self, states: torch.Tensor, context: torch.Tensor,
                return_attn: bool = False, return_sequence: bool = False):
        """Encode a trajectory to a latent.

        return_sequence=False (default): returns h at the *final* step, (B, latent_dim).
        return_sequence=True: returns a per-timestep readout, (B, T, latent_dim) — the
        representation of the trajectory *up to* each step. Used to produce all K JEPA
        targets in a single encoder pass (large CPU saving vs. K windowed passes).
        """
        B, T, _ = states.shape
        nodes = self._embed_nodes(states, context)
        x = nodes.reshape(B * T, STATE_DIM, self.d_model)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, self.mask, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn.reshape(B, T, *attn.shape[1:]))
        x = x.reshape(B, T, STATE_DIM * self.d_model)
        pooled = self.node_pool(x)                      # (B, T, d)
        seq, h_t = self.gru(pooled)                     # seq: (B,T,d), h_t: (1,B,d)
        if return_sequence:
            out = self.readout(seq)                     # (B, T, latent_dim)
        else:
            out = self.readout(h_t.squeeze(0))          # (B, latent_dim)
        if return_attn:
            return out, attn_maps
        return out, None
