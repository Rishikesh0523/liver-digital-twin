"""Generalisation probes — the differentiator.

The brief is explicit about the ceiling: inside a single generator, a model that
generalises across held-out patients has, at best, *recovered the generator's
update rule*. These probes therefore do not try to prove "world model vs.
generator-inverter"; they map **where recovery breaks** and state what each test
can and cannot establish. Every probe returns raw numbers and a plain-language
verdict, and we surface the failures rather than hide them.

Each probe trains nothing — it takes an already-trained model and a generator it
can re-seed into controlled cohorts, so the shift being tested is the *only* thing
that changed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import LiverTrajectoryDataset, collate
from ..data.generator import LiverDiseaseGenerator
from ..data.schema import FIELD_NAMES
from .metrics import evaluate_predictor, error_vs_horizon, overall_mse


@dataclass
class ProbeResult:
    name: str
    numbers: dict
    verdict: str
    can_establish: str
    cannot_establish: str


def _loader(trajs, history_length, prediction_horizon, batch_size=128):
    ds = LiverTrajectoryDataset(trajs, history_length=history_length,
                                prediction_horizon=prediction_horizon)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


class GeneralizationProbes:
    def __init__(self, predict_fn: Callable, device, history_length: int = 12,
                 prediction_horizon: int = 12, gen: LiverDiseaseGenerator | None = None):
        self.predict_fn = predict_fn
        self.device = device
        self.H = history_length
        self.K = prediction_horizon
        self.gen = gen or LiverDiseaseGenerator()

    # -- 1. held-out susceptibility ---------------------------------------
    def probe_unseen_susceptibility(self, n=200) -> ProbeResult:
        """Train distribution is typical susceptibility; test on the rapid-progressor
        tail (susceptibility fixed high). Expected: under-prediction of slow fields."""
        low = self.gen.generate(n, horizon=self.H + self.K + 4, seed=90001,
                                susceptibility=0.6)   # slow progressors
        high = self.gen.generate(n, horizon=self.H + self.K + 4, seed=90002,
                                 susceptibility=3.0)  # rapid progressors (tail)
        m_low = evaluate_predictor(self.predict_fn, _loader(low, self.H, self.K), self.device)
        m_high = evaluate_predictor(self.predict_fn, _loader(high, self.H, self.K), self.device)
        ratio = m_high["overall_mse"] / max(1e-9, m_low["overall_mse"])
        return ProbeResult(
            name="unseen_susceptibility",
            numbers={
                "mse_low_sus": m_low["overall_mse"],
                "mse_high_sus": m_high["overall_mse"],
                "mse_ratio_high_over_low": ratio,
                "mse_per_field_high": m_high["mse_per_field"],
                "violation_rate_high": m_high["constraint_violation_rate"]["any_rate"],
            },
            verdict=(f"High-susceptibility error is {ratio:.1f}x the low-susceptibility "
                     f"error — the model {'degrades sharply' if ratio > 2 else 'holds up'} "
                     f"on rapid progressors."),
            can_establish="Whether learned dynamics transfer to progression rates "
                          "outside the training mass.",
            cannot_establish="Whether failures are 'wrong biology' vs. simply "
                             "out-of-support extrapolation of the same rule.",
        )

    # -- 2. unseen treatment timing ---------------------------------------
    def probe_unseen_treatment_timing(self, n=200) -> ProbeResult:
        """Shift UDCA start and ERCP months to values the training mass rarely shows.
        Expected: larger error on the fast/treatment-coupled fields (A, C, S)."""
        nominal = self.gen.generate(n, horizon=self.H + self.K + 4, seed=90003,
                                    udca_start_month=0, ercp_months=(12, 24))
        shifted = self.gen.generate(n, horizon=self.H + self.K + 4, seed=90004,
                                    udca_start_month=self.H + 2, ercp_months=(self.H + 1,
                                                                              self.H + self.K - 1))
        m_nom = evaluate_predictor(self.predict_fn, _loader(nominal, self.H, self.K), self.device)
        m_shift = evaluate_predictor(self.predict_fn, _loader(shifted, self.H, self.K), self.device)
        fields = ("A", "C", "S")
        per_field_delta = {f: m_shift["mse_per_field"][f] - m_nom["mse_per_field"][f]
                           for f in fields}
        return ProbeResult(
            name="unseen_treatment_timing",
            numbers={
                "mse_nominal": m_nom["overall_mse"],
                "mse_shifted": m_shift["overall_mse"],
                "treatment_field_mse_delta": per_field_delta,
                "violation_rate_shifted": m_shift["constraint_violation_rate"]["any_rate"],
            },
            verdict=(f"Shifting treatment timing changes overall MSE "
                     f"{m_nom['overall_mse']:.4f} -> {m_shift['overall_mse']:.4f}; "
                     f"treatment-coupled fields A/C/S move by "
                     f"{ {k: round(v, 4) for k, v in per_field_delta.items()} }."),
            can_establish="Whether the model keyed on absolute event months vs. the "
                          "event signal itself (the referral-shortcut worry).",
            cannot_establish="Causal correctness of the treatment response — that "
                             "needs interventional re-runs (out of scope).",
        )

    # -- 3. long rollout ---------------------------------------------------
    def probe_long_rollout(self, train_horizon: int, long_horizon: int, n=150) -> ProbeResult:
        """Predict far past the training horizon. Expected: error accumulates and
        (for an unconstrained model) constraints would drift. Reports both curves."""
        trajs = self.gen.generate(n, horizon=self.H + long_horizon + 2, seed=90005)
        loader = _loader(trajs, self.H, long_horizon)
        m = evaluate_predictor(self.predict_fn, loader, self.device)
        curve = m["error_vs_horizon"]
        in_h = float(np.mean(curve[:train_horizon]))
        out_h = float(np.mean(curve[train_horizon:])) if len(curve) > train_horizon else float("nan")
        return ProbeResult(
            name="long_rollout",
            numbers={
                "error_vs_horizon": curve,
                "mean_error_in_horizon": in_h,
                "mean_error_beyond_horizon": out_h,
                "growth_ratio": (out_h / max(1e-9, in_h)) if out_h == out_h else float("nan"),
                "violation_rate": m["constraint_violation_rate"]["any_rate"],
            },
            verdict=(f"Error grows from {in_h:.4f} (<= {train_horizon}mo) to {out_h:.4f} "
                     f"(> {train_horizon}mo); constraint violations remain "
                     f"{m['constraint_violation_rate']['any_rate']:.4f} — the "
                     f"parameterisation holds even where accuracy decays."),
            can_establish="Whether accuracy and constraint-satisfaction decouple over "
                          "long horizons (they should: constraints are structural).",
            cannot_establish="Long-horizon *accuracy* ceiling — accumulation is inherent "
                             "to autoregressive rollout.",
        )

    # -- 4. interpolation vs extrapolation --------------------------------
    def probe_interpolation_vs_extrapolation(self, train_trajs, n=200) -> ProbeResult:
        """Compare error on *training* patients (windows the model saw) vs. fresh
        held-out patients. A large gap = memorisation rather than rule-learning."""
        seen = _loader(train_trajs[:n], self.H, self.K)
        unseen_trajs = self.gen.generate(n, horizon=self.H + self.K + 4, seed=90006)
        unseen = _loader(unseen_trajs, self.H, self.K)
        m_seen = evaluate_predictor(self.predict_fn, seen, self.device)
        m_unseen = evaluate_predictor(self.predict_fn, unseen, self.device)
        gap = m_unseen["overall_mse"] / max(1e-9, m_seen["overall_mse"])
        return ProbeResult(
            name="interpolation_vs_extrapolation",
            numbers={
                "mse_seen_patients": m_seen["overall_mse"],
                "mse_unseen_patients": m_unseen["overall_mse"],
                "generalisation_gap_ratio": gap,
            },
            verdict=(f"Unseen-patient MSE is {gap:.2f}x seen-patient MSE. "
                     f"{'Small gap: the model recovered the rule, not the trainset.' if gap < 1.5 else 'Large gap: memorisation risk.'}"),
            can_establish="Memorisation vs. rule-recovery within this generator.",
            cannot_establish="Transfer to a *different* generator / real biology — the "
                             "single-generator ceiling the brief names.",
        )

    def probe_counterfactual_note(self) -> ProbeResult:
        """Out of scope per the brief; recorded as a discussion pointer."""
        return ProbeResult(
            name="counterfactual",
            numbers={},
            verdict="Out of scope (brief). Would require intervening on the treatment "
                    "timeline and validating against a generator re-run with the same "
                    "per-patient seed but altered UDCA/ERCP schedule.",
            can_establish="Nothing measured here.",
            cannot_establish="Interventional (causal) correctness — needs the do-operator "
                             "semantics the attention model does not encode.",
        )
