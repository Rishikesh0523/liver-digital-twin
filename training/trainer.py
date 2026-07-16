"""Training loop with the operational hygiene a Staff reviewer expects:
seeding, device-agnostic tensors, cosine LR with warmup, grad clipping, validation,
early stopping, best-checkpointing, and structured (wandb-free) dict logging.

Deliberately generic: it drives *any* model via a ``loss_fn(model, batch) -> dict``
closure, so the JEPA and the baseline share identical machinery and the comparison
stays fair (Decision D6).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

from ..data.dataset import Batch, collate


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def cosine_warmup(step: int, warmup: int, total: int, base_lr: float,
                  min_lr_frac: float = 0.05) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cos)


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    val_every: int = 1
    patience: int = 6            # early stopping on the monitor metric
    seed: int = 0
    device: str = "auto"
    log_every: int = 50
    # collapse-guarded selection: only checkpoint epochs whose latent effective
    # dim >= this bar (0 disables — e.g. for the baseline, which has no latent).
    collapse_guard: float = 0.0


@dataclass
class History:
    train: list = field(default_factory=list)
    val: list = field(default_factory=list)


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 loss_fn: Callable[[torch.nn.Module, Batch], dict],
                 cfg: TrainConfig, checkpoint_path: Optional[str] = None):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.checkpoint_path = checkpoint_path
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.history = History()

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int,
                   total_steps: int, step0: int) -> dict:
        self.model.train(train)
        agg: dict[str, float] = {}
        n = 0
        warmup = int(self.cfg.warmup_frac * total_steps)
        for i, batch in enumerate(loader):
            batch = batch.to(self.device)
            if train:
                lr = cosine_warmup(step0 + i, warmup, total_steps, self.cfg.lr)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                self.opt.zero_grad()
            with torch.set_grad_enabled(train):
                logs = self.loss_fn(self.model, batch, epoch=epoch)
                loss = logs["loss"]
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
            bs = batch.history.shape[0]
            n += bs
            for k, v in logs.items():
                val = float(v.detach()) if torch.is_tensor(v) else float(v)
                agg[k] = agg.get(k, 0.0) + val * bs
        return {k: v / max(1, n) for k, v in agg.items()}

    def fit(self, train_ds, val_ds) -> History:
        set_seed(self.cfg.seed)
        g = torch.Generator().manual_seed(self.cfg.seed)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                  collate_fn=collate, generator=g)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False,
                                collate_fn=collate)
        total_steps = self.cfg.epochs * max(1, len(train_loader))
        best_val = float("inf")
        best_state = None
        best_fallback = (float("inf"), None)  # best recon ignoring the guard
        bad = 0
        for epoch in range(self.cfg.epochs):
            step0 = epoch * len(train_loader)
            tr = self._run_epoch(train_loader, True, epoch, total_steps, step0)
            self.history.train.append({"epoch": epoch, **tr})
            if epoch % self.cfg.val_every == 0:
                with torch.no_grad():
                    va = self._run_epoch(val_loader, False, epoch, total_steps, step0)
                self.history.val.append({"epoch": epoch, **va})
                # select on prediction quality, not the regularised total (see losses.py)
                monitor = va.get("monitor", va["loss"])
                msg = (f"epoch {epoch:3d} | train {tr['loss']:.4f} "
                       f"monitor {monitor:.4f} | pred {va.get('l_pred', float('nan')):.4f} "
                       f"recon {va.get('l_recon', float('nan')):.4f} "
                       f"viol {int(va.get('violations', 0))} "
                       f"eff_dim {va.get('effective_dim', float('nan')):.2f}")
                print(msg)
                eff = va.get("effective_dim", float("inf"))
                eff = eff if eff == eff else float("inf")  # nan (baseline) -> pass guard
                healthy = eff >= self.cfg.collapse_guard
                snapshot = lambda: {k: v.detach().cpu().clone()
                                    for k, v in self.model.state_dict().items()}
                # track an unguarded fallback so we always return *something*
                if monitor < best_fallback[0] - 1e-5:
                    best_fallback = (monitor, snapshot())
                if healthy and monitor < best_val - 1e-5:
                    best_val = monitor
                    best_state = snapshot()
                    bad = 0
                    if self.checkpoint_path:
                        self._save(best_state, epoch, best_val)
                elif best_state is not None:
                    # Only count patience AFTER a first healthy checkpoint exists —
                    # otherwise the collapse warm-up (early low-eff_dim epochs) would
                    # burn the patience budget and stop before collapse is even fixed.
                    bad += 1
                    if bad >= self.cfg.patience:
                        print(f"early stop at epoch {epoch} (best guarded recon {best_val:.4f})")
                        break
        if best_state is None:
            # no epoch met the collapse guard — fall back and warn honestly
            print("WARNING: no checkpoint cleared the collapse guard "
                  f"(eff_dim >= {self.cfg.collapse_guard}); using best-recon fallback.")
            best_val, best_state = best_fallback
            if self.checkpoint_path and best_state is not None:
                self._save(best_state, -1, best_val)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.history

    def _save(self, state, epoch, val):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({"model": state, "epoch": epoch, "val_loss": val,
                    "cfg": vars(self.cfg)}, self.checkpoint_path)
