"""Seeded dynamical generator for liver-disease trajectories.

This is the "gift and trap": it emits unlimited labelled trajectories, and a
model that generalises across held-out patients has, at best, recovered *this*
update rule. We therefore keep the generator faithful to the spec and make the
constraints hold *in the data itself* (ratchets clipped at generation time), so
the model is learning a genuinely constraint-respecting process.

Design notes
------------
* Every field is enforced by construction here too — a generated trajectory that
  violated monotonicity would poison the "is the constraint learnable?" question.
* Randomness is fully determined by ``base_seed`` and the patient index, so
  ``generate(seed=k)`` is bit-for-bit reproducible (property-tested).
* Context (disease_class, age, sex, responder, treatment timeline) is *supplied*,
  never predicted, exactly as the brief specifies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .schema import (
    A, C, D, F, FLARE, M, P, S, STATE_DIM, DISEASE_CLASSES, SEXES, upper_bounds,
)

_UPPER = np.asarray(upper_bounds(), dtype=np.float64)


@dataclass
class PatientContext:
    """Immutable per-patient context constants (not predicted by the model)."""

    disease_class: str
    age: float                       # years
    sex: str                         # "F" | "M"
    responder: int                   # 0 | 1
    udca_start_month: Optional[int]  # month UDCA begins, or None
    ercp_months: tuple               # months at which an ERCP occurs
    # latent per-patient constants that drive the dynamics
    susceptibility: float
    hazard_rate: float
    baseline_A: float
    baseline_C: float

    def as_feature_dict(self) -> dict:
        return {
            "disease_class": self.disease_class,
            "age": self.age,
            "sex": self.sex,
            "responder": self.responder,
            "udca_start_month": self.udca_start_month,
            "ercp_months": list(self.ercp_months),
            "susceptibility": self.susceptibility,
            "hazard_rate": self.hazard_rate,
        }


@dataclass
class Trajectory:
    """One patient's trajectory: states over time plus context."""

    states: np.ndarray          # (T, 8) float64, monthly
    context: PatientContext
    ercp_mask: np.ndarray       # (T,) bool, True at ERCP months
    udca_mask: np.ndarray       # (T,) bool, True while UDCA active
    patient_id: int = field(default=-1)

    def __len__(self) -> int:
        return self.states.shape[0]


# Per-disease baseline modifiers. PSC: stricture/flare-driven (biliary).
# PBC: cholestatic, UDCA-responsive. AIH: inflammation-driven.
_DISEASE_MODS = {
    "psc": dict(sus=1.15, a0=0.05, c0=0.05, stricture=1.0, flare=1.3),
    "pbc": dict(sus=0.90, a0=0.00, c0=0.10, stricture=0.3, flare=0.6),
    "aih": dict(sus=1.05, a0=0.12, c0=0.02, stricture=0.4, flare=0.9),
}


class LiverDiseaseGenerator:
    """Produce clinically plausible 8-D liver-disease trajectories.

    Dynamics (dt = 1 month):
        F  += susceptibility * (A + C) * 0.02 * dt         (+noise), ratchet
        D  += susceptibility * C * 0.015 * dt              (+noise), ratchet
        S  += A * 0.1 * dt  - ERCP_relief*I[ercp]          , ratchet-up / ERCP step-down
        P  += F * 0.05 * dt                                (+noise), ratchet
        A   = mean_revert(A, mu, tau=3) + flare_impact*I[flare]
        C   = mean_revert(C, mu, tau=2) + 0.8*flare_impact*I[flare]
              - treatment_suppression * responder * I[udca_active]
        M  += hazard_rate * F * C * dt                     , ratchet, [0,2]
        flare = flare*decay + Poisson_trigger(A, C)        , transient
    """

    def __init__(
        self,
        dt: float = 1.0,
        # coupling / rate constants (kept explicit so the memo can reference them)
        k_fibrosis: float = 0.02,
        k_ductopenia: float = 0.015,
        k_stricture: float = 0.10,
        k_portal: float = 0.05,
        ercp_relief: float = 0.35,
        flare_impact: float = 0.45,
        treatment_suppression: float = 0.20,
        flare_decay: float = 0.55,
        noise_scale: float = 0.004,
        max_flare_rate: float = 0.25,
    ) -> None:
        self.dt = dt
        self.k_fibrosis = k_fibrosis
        self.k_ductopenia = k_ductopenia
        self.k_stricture = k_stricture
        self.k_portal = k_portal
        self.ercp_relief = ercp_relief
        self.flare_impact = flare_impact
        self.treatment_suppression = treatment_suppression
        self.flare_decay = flare_decay
        self.noise_scale = noise_scale
        self.max_flare_rate = max_flare_rate

    # -- sampling of per-patient context ----------------------------------
    def sample_context(self, rng: np.random.Generator,
                       disease_class: Optional[str] = None,
                       responder: Optional[int] = None,
                       susceptibility: Optional[float] = None,
                       udca_start_month: Optional[int] = None,
                       ercp_months: Optional[tuple] = None,
                       horizon: int = 36) -> PatientContext:
        dc = disease_class or rng.choice(DISEASE_CLASSES)
        mod = _DISEASE_MODS[dc]
        sus = susceptibility if susceptibility is not None else \
            float(rng.lognormal(mean=0.0, sigma=0.5)) * mod["sus"]
        hazard = float(rng.uniform(0.001, 0.01))
        b_a = float(rng.beta(2, 5)) + mod["a0"]
        b_c = float(rng.beta(2, 5)) + mod["c0"]
        age = float(np.clip(rng.normal(50, 12), 18, 85))
        sex = str(rng.choice(SEXES))
        resp = responder if responder is not None else int(rng.random() < 0.5)
        # age slightly increases susceptibility
        sus = sus * (1.0 + 0.004 * (age - 50))
        if udca_start_month is None and rng.random() < 0.8:
            udca_start_month = int(rng.integers(0, max(1, horizon // 3)))
        if ercp_months is None:
            n_ercp = int(rng.integers(0, 3)) if mod["stricture"] > 0.5 else 0
            ercp_months = tuple(sorted(int(m) for m in
                                       rng.choice(np.arange(3, horizon), size=n_ercp,
                                                  replace=False))) if n_ercp else tuple()
        return PatientContext(
            disease_class=dc, age=age, sex=sex, responder=resp,
            udca_start_month=udca_start_month, ercp_months=tuple(ercp_months),
            susceptibility=sus, hazard_rate=hazard,
            baseline_A=float(np.clip(b_a, 0, 1)), baseline_C=float(np.clip(b_c, 0, 1)),
        )

    # -- single trajectory rollout ----------------------------------------
    def rollout(self, ctx: PatientContext, horizon: int,
                rng: np.random.Generator) -> Trajectory:
        mod = _DISEASE_MODS[ctx.disease_class]
        T = horizon
        x = np.zeros((T, STATE_DIM), dtype=np.float64)
        ercp_set = set(int(m) for m in ctx.ercp_months)
        # ercp_mask[m] == True iff month m is an ERCP month. Convention (D2):
        # S relief is *destination-gated* — the step-down appears AT the ERCP
        # month, so the drop from x[m-1] to x[m] is legal iff month m is ERCP.
        ercp_mask = np.array([m in ercp_set for m in range(T)], dtype=bool)
        udca_mask = np.zeros(T, dtype=bool)

        # initial state: slow fields near zero, fast fields at baseline
        x[0, A] = ctx.baseline_A
        x[0, C] = ctx.baseline_C
        x[0, F] = float(rng.uniform(0.0, 0.05))
        x[0, D] = 0.0
        x[0, S] = 0.0
        x[0, P] = float(rng.uniform(0.0, 0.03))
        x[0, M] = 0.0
        x[0, FLARE] = 0.0

        mu_A = 0.10
        mu_C = 0.15

        for t in range(T - 1):
            cur = x[t]
            udca_active = (ctx.udca_start_month is not None
                           and t >= ctx.udca_start_month)
            udca_mask[t] = udca_active
            ercp_dest = (t + 1) in ercp_set   # month t+1 is an ERCP month
            nxt = cur.copy()

            eps = rng.normal(0.0, self.noise_scale, size=STATE_DIM)

            # --- flare first: transient decay + stochastic trigger driven by A,C.
            # Computing it before A/C lets a fresh flare perturb A and C *together*
            # at the same step (the coupling the spec calls out), then all decay.
            trigger_rate = self.max_flare_rate * mod["flare"] * (0.5 * cur[A] + 0.5 * cur[C])
            new_flare = float(rng.random() < trigger_rate) * float(rng.uniform(0.4, 1.0))
            flare_next = float(np.clip(cur[FLARE] * self.flare_decay + new_flare, 0.0, 1.0))
            nxt[FLARE] = flare_next

            # --- fast, mean-reverting fields, jointly perturbed by the new flare
            a = cur[A] + (mu_A - cur[A]) / 3.0 + self.flare_impact * flare_next + eps[A]
            c = (cur[C] + (mu_C - cur[C]) / 2.0
                 + 0.8 * self.flare_impact * flare_next
                 - self.treatment_suppression * ctx.responder * float(udca_active)
                 + eps[C])
            nxt[A] = np.clip(a, 0.0, 1.0)
            nxt[C] = np.clip(c, 0.0, 1.0)

            # --- ratchet slow fields (driven by *current* A,C,F) -------------
            dF = ctx.susceptibility * (cur[A] + cur[C]) * self.k_fibrosis * self.dt + eps[F]
            nxt[F] = np.clip(cur[F] + max(dF, 0.0), cur[F], 1.0)  # ratchet by construction

            dD = ctx.susceptibility * cur[C] * self.k_ductopenia * self.dt + max(eps[D], 0.0)
            nxt[D] = np.clip(cur[D] + max(dD, 0.0), cur[D], 1.0)

            dP = cur[F] * self.k_portal * self.dt + eps[P]
            nxt[P] = np.clip(cur[P] + max(dP, 0.0), cur[P], 1.0)

            # --- S: ratchet up with A, step DOWN only at ERCP (decision D2) ---
            dS = cur[A] * self.k_stricture * mod["stricture"] * self.dt
            s_up = np.clip(cur[S] + max(dS, 0.0), cur[S], 1.0)
            if ercp_dest:  # relief lands at the ERCP month (destination-gated)
                nxt[S] = float(np.clip(s_up - self.ercp_relief, 0.0, 1.0))
            else:
                nxt[S] = s_up

            # --- M: hazard of sustained F*C, ratchet, [0,2] ------------------
            dM = ctx.hazard_rate * cur[F] * cur[C] * self.dt
            nxt[M] = np.clip(cur[M] + max(dM, 0.0), cur[M], _UPPER[M])

            x[t + 1] = nxt

        # record UDCA status for the final month too (ercp_mask is precomputed)
        last = T - 1
        udca_mask[last] = (ctx.udca_start_month is not None and last >= ctx.udca_start_month)
        return Trajectory(states=x, context=ctx, ercp_mask=ercp_mask, udca_mask=udca_mask)

    # -- cohort generation -------------------------------------------------
    def generate(self, n_patients: int, horizon: int = 36, seed: int = 0,
                 **context_overrides) -> list[Trajectory]:
        """Generate ``n_patients`` reproducible trajectories.

        ``context_overrides`` (e.g. ``disease_class="psc"``, ``responder=1``,
        ``susceptibility=2.0``) are forwarded to :meth:`sample_context`, enabling
        the generalisation probes to carve controlled cohorts.
        """
        trajectories = []
        for i in range(n_patients):
            rng = np.random.default_rng([seed, i])  # per-patient stream -> reproducible
            ctx = self.sample_context(rng, horizon=horizon, **context_overrides)
            traj = self.rollout(ctx, horizon=horizon, rng=rng)
            traj.patient_id = i
            trajectories.append(traj)
        return trajectories
