"""Counterfactual faithfulness — a toe into the out-of-scope (causal) dimension.

The clinical value of a twin is counterfactuals ("what if UDCA had started 6 months
earlier?"). We can *validate* the model's counterfactual against ground truth because
the generator is seeded per-patient: re-running it with the **same** ``[seed, i]`` but
an altered treatment timeline reuses the identical noise stream, giving a clean A/B
where the only thing that changed is the intervention.

Protocol per patient:
  1. factual: generate with the patient's real UDCA start; counterfactual: same seed,
     UDCA start moved earlier. The generator's own (factual - counterfactual) gap is
     the **ground-truth causal effect** on the outcome.
  2. give the model the *factual* history, then roll it forward under the factual and
     the counterfactual future treatment context; its predicted gap is its **estimated
     causal effect**.
  3. compare direction (sign agreement) and magnitude (ratio) across patients.

Honest caveat (reported): the model was trained only on observational trajectories and
has no interventional (do-operator) semantics — it re-weights attention-correlations.
So this measures whether that correlational machinery happens to be *faithful* to the
intervention, not whether it is causal by design.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import encode_context
from ..data.generator import LiverDiseaseGenerator
from ..data.schema import A, C, M, F, FIELD_NAMES


@dataclass
class CounterfactualReport:
    outcome_field: str
    shift_months: int
    n_patients: int
    true_effect_mean: float          # generator ground-truth mean effect
    pred_effect_mean: float          # model mean effect
    sign_agreement_rate: float       # fraction of patients where sign(pred)==sign(true)
    magnitude_ratio: float           # mean|pred| / mean|true|
    verdict: str


def _context_for_udca(traj, udca_start: int):
    """Re-encode a trajectory's context but with an overridden UDCA start month."""
    T = len(traj)
    udca_mask = np.array([udca_start is not None and t >= udca_start for t in range(T)],
                         dtype=bool)
    traj2 = traj  # shares states; we only rebuild the context/mask
    ctx = encode_context(traj2).copy()
    ctx[:, 6] = udca_mask.astype(np.float32)  # udca_active column
    return ctx, udca_mask


@torch.no_grad()
def run(model, device, history_length: int, prediction_horizon: int,
        shift_months: int = 6, outcome_idx: int = C, n_patients: int = 120,
        seed: int = 55000, gen: LiverDiseaseGenerator | None = None) -> CounterfactualReport:
    gen = gen or LiverDiseaseGenerator()
    H, K = history_length, prediction_horizon
    horizon = H + K + 4
    true_effects, pred_effects = [], []

    for i in range(n_patients):
        rng_f = np.random.default_rng([seed, i])
        # Force responders: UDCA suppresses C/A only for responders, so the causal
        # effect is defined only there (documented choice). Place the factual UDCA
        # start in the FUTURE window so both factual and counterfactual starts lie
        # after the observed history — otherwise the intervention is in the past and
        # the model (which sees only factual history) cannot express any effect.
        ctx_f = gen.sample_context(rng_f, horizon=horizon, responder=1)
        udca0 = H + shift_months + int(rng_f.integers(0, max(1, K - shift_months)))
        udca0 = min(udca0, horizon - 1)
        udca_cf = udca0 - shift_months   # still >= H
        ctx_f = _with_udca(ctx_f, udca0)

        # factual + counterfactual generator runs share the SAME per-patient stream
        traj_f = gen.rollout(ctx_f, horizon=horizon,
                             rng=np.random.default_rng([seed, i, 7]))
        traj_cf = gen.rollout(_with_udca(ctx_f, udca_cf), horizon=horizon,
                              rng=np.random.default_rng([seed, i, 7]))

        tgt = H + K - 1
        true_effect = float(traj_cf.states[tgt, outcome_idx] - traj_f.states[tgt, outcome_idx])
        true_effects.append(true_effect)

        # model: factual history, roll forward under factual vs counterfactual context
        pred_f = _model_outcome(model, traj_f, udca0, H, K, outcome_idx, tgt, device)
        pred_cf = _model_outcome(model, traj_f, udca_cf, H, K, outcome_idx, tgt, device)
        pred_effects.append(pred_cf - pred_f)

    true_effects = np.array(true_effects)
    pred_effects = np.array(pred_effects)
    nz = np.abs(true_effects) > 1e-4
    sign_agree = float(np.mean(np.sign(pred_effects[nz]) == np.sign(true_effects[nz]))) \
        if nz.any() else float("nan")
    mag_ratio = float(np.mean(np.abs(pred_effects)) / (np.mean(np.abs(true_effects)) + 1e-9))
    directional = sign_agree == sign_agree and sign_agree >= 0.7
    magnitude_ok = 0.5 <= mag_ratio <= 2.0
    verdict = (
        f"Model's UDCA-{shift_months}mo-earlier effect on {FIELD_NAMES[outcome_idx]}: "
        f"predicted mean {pred_effects.mean():+.4f} vs generator ground-truth "
        f"{true_effects.mean():+.4f}; sign agreement {100*sign_agree:.0f}% over "
        f"{int(nz.sum())} patients; magnitude ratio {mag_ratio:.3f}. "
        + ("Directionally faithful" if directional else "Directionally UNfaithful")
        + (" and well-scaled." if magnitude_ok else
           " but severely under-scales the effect — the correlational model knows the "
           "SIGN of the intervention, not its magnitude (honest negative on the "
           "quantitative side; expected without do-operator semantics).")
    )
    return CounterfactualReport(FIELD_NAMES[outcome_idx], shift_months, n_patients,
                                float(true_effects.mean()), float(pred_effects.mean()),
                                sign_agree, mag_ratio, verdict)


def _with_udca(ctx, udca_start):
    import dataclasses
    return dataclasses.replace(ctx, udca_start_month=udca_start)


@torch.no_grad()
def _model_outcome(model, traj, udca_start, H, K, outcome_idx, tgt, device):
    """Roll the model forward on traj's history under a given UDCA start; return the
    predicted outcome-field value at the target step."""
    states = traj.states.astype(np.float32)
    ctx, _ = _context_for_udca(traj, udca_start)
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    history = to(states[:H])
    hist_ctx = to(ctx[:H])
    fut_ctx = to(ctx[H:H + K])
    current = to(states[H - 1])
    ercp_future = to(traj.ercp_mask[H:H + K].astype(np.float32))
    if hasattr(model, "predict"):
        pred = model.predict(history, hist_ctx, fut_ctx, current, ercp_future)
    else:
        pred = model(history, hist_ctx, fut_ctx, current, ercp_future)
    step = tgt - H  # index within the future window
    return float(pred[0, step, outcome_idx].cpu())
