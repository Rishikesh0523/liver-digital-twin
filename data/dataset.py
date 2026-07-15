"""Torch dataset over generator trajectories.

Splits are **by patient** (never leak a patient across train/val/test), because
the whole generalisation question is "does it transfer to unseen patients?".
Each item is a (history, context, future, current, meta) tuple; a JEPA sample
predicts ``future`` from ``history`` conditioned on time-varying ``context``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .generator import LiverDiseaseGenerator, Trajectory
from .schema import (
    CONTEXT_DIM, DISEASE_CLASSES, STATE_DIM,
)


def encode_context(traj: Trajectory) -> np.ndarray:
    """Encode context as a (T, CONTEXT_DIM) array (time-varying for treatment).

    Layout: [disease_onehot(3), age_norm, sex, responder, udca_active, ercp_now].
    Static fields are broadcast across time; udca_active/ercp_now vary with t.
    """
    ctx = traj.context
    T = len(traj)
    onehot = np.zeros(3, dtype=np.float32)
    onehot[DISEASE_CLASSES.index(ctx.disease_class)] = 1.0
    age_norm = np.float32((ctx.age - 50.0) / 20.0)
    sex = np.float32(1.0 if ctx.sex == "M" else 0.0)
    resp = np.float32(ctx.responder)
    static = np.concatenate([onehot, [age_norm, sex, resp]]).astype(np.float32)
    out = np.zeros((T, CONTEXT_DIM), dtype=np.float32)
    out[:, :6] = static
    out[:, 6] = traj.udca_mask.astype(np.float32)
    out[:, 7] = traj.ercp_mask.astype(np.float32)
    return out


@dataclass
class Batch:
    """A collated batch (tensors already on the default device at use time)."""

    history: torch.Tensor        # (B, H, STATE_DIM)
    hist_context: torch.Tensor   # (B, H, CONTEXT_DIM)
    future: torch.Tensor         # (B, K, STATE_DIM)
    fut_context: torch.Tensor    # (B, K, CONTEXT_DIM)
    current: torch.Tensor        # (B, STATE_DIM) — last observed state (ratchet base)
    ercp_future: torch.Tensor    # (B, K) bool — ERCP flags over the future window
    patient_id: torch.Tensor     # (B,)

    def to(self, device) -> "Batch":
        return Batch(
            self.history.to(device), self.hist_context.to(device),
            self.future.to(device), self.fut_context.to(device),
            self.current.to(device), self.ercp_future.to(device),
            self.patient_id.to(device),
        )


class LiverTrajectoryDataset(Dataset):
    """Sliding-window (history -> future) samples over patient trajectories."""

    def __init__(self, trajectories: list[Trajectory], history_length: int = 12,
                 prediction_horizon: int = 12, stride: int = 1):
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.samples: list[tuple[int, int]] = []  # (traj_index, window_start)
        self.trajectories = trajectories
        for ti, traj in enumerate(trajectories):
            T = len(traj)
            last_start = T - (history_length + prediction_horizon)
            for s in range(0, max(0, last_start) + 1, stride):
                self.samples.append((ti, s))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        ti, s = self.samples[i]
        traj = self.trajectories[ti]
        H, K = self.history_length, self.prediction_horizon
        states = traj.states.astype(np.float32)
        ctx = encode_context(traj)
        history = states[s:s + H]
        future = states[s + H:s + H + K]
        hist_ctx = ctx[s:s + H]
        fut_ctx = ctx[s + H:s + H + K]
        current = states[s + H - 1]
        ercp_future = traj.ercp_mask[s + H:s + H + K].astype(np.float32)
        return {
            "history": torch.from_numpy(history),
            "hist_context": torch.from_numpy(hist_ctx),
            "future": torch.from_numpy(future),
            "fut_context": torch.from_numpy(fut_ctx),
            "current": torch.from_numpy(current),
            "ercp_future": torch.from_numpy(ercp_future),
            "patient_id": torch.tensor(traj.patient_id, dtype=torch.long),
        }


def collate(items: list[dict]) -> Batch:
    stack = lambda k: torch.stack([it[k] for it in items])
    return Batch(
        history=stack("history"), hist_context=stack("hist_context"),
        future=stack("future"), fut_context=stack("fut_context"),
        current=stack("current"), ercp_future=stack("ercp_future"),
        patient_id=stack("patient_id"),
    )


def make_splits(
    n_train: int, n_val: int, n_test: int, horizon: int = 36,
    history_length: int = 12, prediction_horizon: int = 12,
    seed: int = 0, generator: Optional[LiverDiseaseGenerator] = None,
    train_stride: int = 1, **context_overrides,
) -> dict[str, LiverTrajectoryDataset]:
    """Build train/val/test datasets from disjoint patient seeds (no leakage).

    ``train_stride`` > 1 subsamples the (heavily overlapping) training windows for
    speed; val/test stay at stride 1 for accurate metrics.
    """
    gen = generator or LiverDiseaseGenerator()
    # disjoint seeds -> disjoint patient populations
    train = gen.generate(n_train, horizon=horizon, seed=seed, **context_overrides)
    val = gen.generate(n_val, horizon=horizon, seed=seed + 10_000, **context_overrides)
    test = gen.generate(n_test, horizon=horizon, seed=seed + 20_000, **context_overrides)
    kw = dict(history_length=history_length, prediction_horizon=prediction_horizon)
    return {
        "train": LiverTrajectoryDataset(train, stride=train_stride, **kw),
        "val": LiverTrajectoryDataset(val, **kw),
        "test": LiverTrajectoryDataset(test, **kw),
        "_raw": {"train": train, "val": val, "test": test},
    }
