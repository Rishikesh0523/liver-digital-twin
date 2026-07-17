"""Train the JEPA world model and the baseline, save both checkpoints.

    python -m liver_world_model.experiments.train --config liver_world_model/configs/default.yaml

Runs end-to-end with no manual steps: generates data, trains both models, writes
checkpoints. Device-agnostic; seeded for reproducibility.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import yaml

from ..data.dataset import make_splits
from ..models.baseline import DirectPredictor
from ..models.jepa import LiverJEPA
from ..training.losses import LossWeights
from ..training.objectives import make_baseline_objective, make_jepa_objective
from ..training.trainer import Trainer, TrainConfig, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_datasets(cfg: dict):
    d = cfg["data"]
    return make_splits(
        n_train=d["n_patients_train"], n_val=d["n_patients_val"],
        n_test=d["n_patients_test"], horizon=d["horizon"],
        history_length=d["history_length"], prediction_horizon=d["prediction_horizon"],
        seed=d["seed"], train_stride=d.get("train_stride", 1),
    )


def build_train_config(cfg: dict) -> TrainConfig:
    t = cfg["training"]
    return TrainConfig(
        epochs=t["epochs"], batch_size=t["batch_size"], lr=t["lr"],
        weight_decay=t["weight_decay"], warmup_frac=t["warmup_frac"],
        grad_clip=t["grad_clip"], patience=t["patience"], seed=t["seed"],
        device=t["device"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="liver_world_model/configs/default.yaml")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    m = cfg["model"]
    d = cfg["data"]
    splits = build_datasets(cfg)
    tcfg = build_train_config(cfg)

    print("=" * 60, "\nTraining JEPA world model\n", "=" * 60, sep="")
    jepa = LiverJEPA(
        latent_dim=m["latent_dim"], d_model=m["d_model"], n_heads=m["n_attention_heads"],
        n_attn_layers=m["n_attention_layers"], predictor_hidden=m["predictor_hidden"],
        decoder_hidden=tuple(m["decoder_hidden"]), history_length=d["history_length"],
        dt=m["dt"],
    )
    base_w = LossWeights(cfg["training"]["w_pred"], cfg["training"]["w_constr"],
                         cfg["training"]["w_collapse"], cfg["training"]["w_recon"])
    # JEPA selection is collapse-guarded: only checkpoint epochs whose latent is
    # healthy (eff_dim >= threshold). The baseline has no latent, so guard=0.
    jepa_tcfg = dataclasses.replace(
        tcfg, collapse_guard=cfg["evaluation"]["collapse_alert_threshold"])
    jepa_hist = Trainer(jepa, make_jepa_objective(base_w), jepa_tcfg,
                        checkpoint_path=os.path.join(ckpt_dir, "jepa_best.pt")).fit(
        splits["train"], splits["val"])

    print("\n", "=" * 60, "\nTraining baseline (direct predictor)\n", "=" * 60, sep="")
    baseline = DirectPredictor(dt=m["dt"])
    base_hist = Trainer(baseline, make_baseline_objective(), tcfg,
                        checkpoint_path=os.path.join(ckpt_dir, "baseline_best.pt")).fit(
        splits["train"], splits["val"])

    # persist training histories (for the loss/collapse-curve figures)
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "train_history.json"), "w") as f:
        json.dump({"jepa": {"train": jepa_hist.train, "val": jepa_hist.val},
                   "baseline": {"train": base_hist.train, "val": base_hist.val}}, f, indent=2)
    print("\nDone. Checkpoints written to", ckpt_dir)


if __name__ == "__main__":
    main()
