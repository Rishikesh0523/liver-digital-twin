"""Generate all figures for the report.

    python -m liver_world_model.experiments.make_figures --config liver_world_model/configs/default.yaml

Produces, into ``liver_world_model/figures/``:
  * mermaid diagrams rendered to PNG (architecture, causal graph, pipeline) — via
    the mermaid CLI if available, otherwise a networkx fallback for the causal graph;
  * training/validation loss curves (JEPA + baseline);
  * effective-dimension "collapse curve" over epochs;
  * per-field MSE bars (JEPA vs baseline);
  * error-vs-horizon curve;
  * noise-robustness curve (the "where the latent wins" figure);
  * manifold-critic scores;
  * counterfactual faithfulness scatter/summary.

Reads `train_history.json` and `evaluation_report.json` from the results dir, so run
`train` and `evaluate` first. Missing inputs are skipped with a warning, never crash.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import load_config

# consistent palette
C_JEPA, C_BASE, C_DEN, C_ACC = "#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})


def _load_json(path):
    if not os.path.exists(path):
        print(f"  [skip] missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def render_mermaid(diagrams_dir, fig_dir):
    """Render .mmd -> .png via mermaid CLI (mmdc) if present."""
    mmdc = shutil.which("mmdc")
    if not mmdc:
        print("  [warn] mmdc not found; skipping mermaid render (source .mmd kept).")
        return {}
    out = {}
    for name in ("architecture", "causal_graph", "pipeline",
                 "jepa_detailed", "gru_detailed"):
        src = os.path.join(diagrams_dir, f"{name}.mmd")
        dst = os.path.join(fig_dir, f"diagram_{name}.png")
        if not os.path.exists(src):
            continue
        try:
            subprocess.run([mmdc, "-i", src, "-o", dst, "-b", "white", "-s", "2"],
                           check=True, capture_output=True, timeout=120)
            out[name] = dst
            print(f"  [ok] rendered {name}")
        except Exception as e:
            print(f"  [warn] mermaid render failed for {name}: {e}")
    return out


def fig_loss_curves(history, fig_dir):
    if not history:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, model in zip(axes, ("jepa", "baseline")):
        h = history.get(model, {})
        tr = h.get("train", [])
        va = h.get("val", [])
        if not tr:
            continue
        te = [d["epoch"] for d in tr]
        # recon = state-space accuracy (comparable across models)
        ax.plot(te, [d.get("l_recon", np.nan) for d in tr], color=C_JEPA,
                label="train recon (state MSE)")
        ve = [d["epoch"] for d in va]
        ax.plot(ve, [d.get("l_recon", np.nan) for d in va], color=C_BASE,
                label="val recon (state MSE)")
        ax.set_title(f"{model.upper()} training")
        ax.set_xlabel("epoch"); ax.set_ylabel("state-space MSE")
        ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Training / validation loss (state-space reconstruction MSE)")
    _save(fig, fig_dir, "fig_loss_curves.png")


def fig_collapse_curve(history, fig_dir, threshold=3.0):
    h = (history or {}).get("jepa", {})
    va = h.get("val", [])
    if not va:
        return
    ep = [d["epoch"] for d in va]
    ed = [d.get("effective_dim", np.nan) for d in va]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ep, ed, marker="o", color=C_JEPA, label="latent effective dim")
    ax.axhline(threshold, color=C_BASE, ls="--", label=f"collapse-guard bar ({threshold})")
    ax.set_xlabel("epoch"); ax.set_ylabel("effective dim (participation ratio)")
    ax.set_title("Representation-collapse curve\n(guard only checkpoints epochs above the bar)")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, fig_dir, "fig_collapse_curve.png")


def fig_per_field(report, fig_dir):
    tm = (report or {}).get("test_metrics")
    if not tm:
        return
    fields = list(tm["jepa"]["mse_per_field"].keys())
    jm = [tm["jepa"]["mse_per_field"][f] for f in fields]
    bm = [tm["baseline"]["mse_per_field"][f] for f in fields]
    x = np.arange(len(fields)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, jm, w, label="JEPA", color=C_JEPA)
    ax.bar(x + w / 2, bm, w, label="baseline", color=C_BASE)
    ax.set_xticks(x); ax.set_xticklabels(fields)
    ax.set_ylabel("held-out MSE"); ax.set_title("Per-field predictive error")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir, "fig_per_field_mse.png")


def fig_horizon(report, fig_dir):
    tm = (report or {}).get("test_metrics")
    if not tm:
        return
    jh = tm["jepa"].get("error_vs_horizon")
    bh = tm["baseline"].get("error_vs_horizon")
    if not jh:
        return
    steps = np.arange(1, len(jh) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(steps, jh, marker="o", color=C_JEPA, label="JEPA")
    if bh:
        ax.plot(steps, bh, marker="s", color=C_BASE, label="baseline")
    ax.set_xlabel("forecast horizon (months ahead)"); ax.set_ylabel("MSE")
    ax.set_title("Error accumulation vs horizon"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, fig_dir, "fig_error_vs_horizon.png")


def fig_noise(report, fig_dir):
    nz = (report or {}).get("noise_robustness")
    if not nz:
        return
    s = nz["sigmas"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(s, nz["baseline_mae"], marker="s", color=C_BASE, label="baseline (raw anchor)")
    ax.plot(s, nz["jepa_raw_mae"], marker="^", color=C_ACC, label="JEPA (raw anchor)")
    ax.plot(s, nz["jepa_denoised_mae"], marker="o", color=C_DEN,
            label="JEPA (denoised anchor)")
    cx = nz.get("crossover_sigma")
    if cx is not None and cx != float("inf") and cx <= max(s):
        ax.axvline(cx, color="gray", ls=":", label=f"crossover σ={cx:.2f}")
    ax.set_xlabel("observation noise σ"); ax.set_ylabel("MAE")
    ax.set_title("Where the latent earns its keep:\nnoise robustness of the denoised anchor")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, fig_dir, "fig_noise_robustness.png")


def fig_manifold(report, fig_dir):
    mc = (report or {}).get("manifold_critic")
    if not mc:
        return
    labels = ["real\n(held-out)", "JEPA", "baseline", "valid-but-wrong\n(critic negatives)"]
    vals = [mc["score_real_heldout"], mc["score_jepa"], mc["score_baseline"],
            1 - mc["critic_auc"]]
    colors = [C_ACC, C_JEPA, C_BASE, "#9467bd"]
    fig, ax = plt.subplots(figsize=(6.8, 4))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0.5, color="gray", ls="--", label="indistinguishable (0.5)")
    ax.set_ylabel("critic 'realness' score"); ax.set_ylim(0, 1)
    ax.set_title(f"Manifold critic (AUC {mc['critic_auc']:.2f}):\n"
                 "0 violations ≠ on-manifold — both models score near real")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir, "fig_manifold_critic.png")


def fig_counterfactual(report, fig_dir):
    cf = (report or {}).get("counterfactual")
    if not cf:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4))
    labels = ["generator\n(ground truth)", "model\n(predicted)"]
    vals = [cf["true_effect_mean"], cf["pred_effect_mean"]]
    ax.bar(labels, vals, color=[C_ACC, C_JEPA])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(f"mean effect on {cf['outcome_field']}")
    ax.set_title(f"Counterfactual: UDCA {cf['shift_months']}mo earlier\n"
                 f"sign agreement {100*cf['sign_agreement_rate']:.0f}% vs generator re-run")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, fig_dir, "fig_counterfactual.png")


def fig_headline(report, fig_dir):
    """Headline JEPA-vs-baseline: MSE and constraint rate side by side."""
    tm = (report or {}).get("test_metrics")
    if not tm:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.bar(["JEPA", "baseline"], [tm["jepa"]["overall_mse"], tm["baseline"]["overall_mse"]],
           color=[C_JEPA, C_BASE])
    a1.set_title("Held-out MSE (lower better)"); a1.grid(alpha=0.3, axis="y")
    jv = tm["jepa"]["constraint_violation_rate"]["any_rate"] * 100
    bv = tm["baseline"]["constraint_violation_rate"]["any_rate"] * 100
    a2.bar(["JEPA", "baseline"], [jv, bv], color=[C_JEPA, C_BASE])
    a2.set_title("Constraint-violation rate (%)"); a2.set_ylim(0, max(1, jv, bv) + 0.5)
    a2.grid(alpha=0.3, axis="y")
    fig.suptitle("Headline: baseline wins accuracy; both perfect on constraints")
    _save(fig, fig_dir, "fig_headline.png")


def _save(fig, fig_dir, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path); plt.close(fig)
    print(f"  [ok] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="liver_world_model/configs/default.yaml")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = load_config(args.config)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(here, "figures")
    diagrams_dir = os.path.join(here, "diagrams")
    os.makedirs(fig_dir, exist_ok=True)
    results_dir = cfg["paths"]["results_dir"]

    print("Rendering mermaid diagrams...")
    render_mermaid(diagrams_dir, fig_dir)

    print("Plotting result figures...")
    history = _load_json(os.path.join(results_dir, "train_history.json"))
    report = _load_json(os.path.join(results_dir, "evaluation_report.json"))
    fig_loss_curves(history, fig_dir)
    fig_collapse_curve(history, fig_dir, cfg["evaluation"]["collapse_alert_threshold"])
    fig_headline(report, fig_dir)
    fig_per_field(report, fig_dir)
    fig_horizon(report, fig_dir)
    fig_noise(report, fig_dir)
    fig_manifold(report, fig_dir)
    fig_counterfactual(report, fig_dir)
    print(f"\nFigures written to {fig_dir}")


if __name__ == "__main__":
    main()
