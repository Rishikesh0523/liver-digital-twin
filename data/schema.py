"""Frozen state-vector schema shared by every module.

The clinical state ``x(t) in R^8`` is the single organizing construct (generator
output, world-model decode target, and audit record). Every other file imports
field indices and metadata from here so there are no magic numbers anywhere.

Decision D5: field order is fixed as F, D, S, P, A, C, M, flare.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Field indices (immutable contract) -----------------------------------
F = 0      # fibrosis                 ratchet, non-decreasing
D = 1      # ductopenia               ratchet, irreversible
S = 2      # biliary strictures       ratchet up, step-down only at ERCP
P = 3      # portal hypertension      ratchet, non-decreasing
A = 4      # inflammatory activity    fast, mean-reverting
C = 5      # cholestasis              fast, with flares
M = 6      # malignancy hazard        monotone non-decreasing accumulator, [0,2]
FLARE = 7  # acute cholangitis flare  transient, decays

STATE_DIM = 8
FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]

# --- Field groupings by dynamical behaviour -------------------------------
# Ratchet-up fields: guaranteed non-decreasing by construction.
RATCHET_UP = (F, D, P, M)
# Bounded free fields (fast dynamics), sigmoid-parameterised in [0,1].
FREE = (A, C, FLARE)
# S is special: ratchet-up except a downward step is permitted at ERCP months.
S_IDX = S

# --- Per-field upper bounds (lower bound is 0 for all) --------------------
UPPER = [1.0] * STATE_DIM
UPPER[M] = 2.0  # malignancy hazard accumulator lives in [0, 2]


def upper_bounds():
    """Return per-field upper bounds as a plain list (M is 2.0, rest 1.0)."""
    return list(UPPER)


# --- Context schema (constants supplied alongside, never predicted) -------
DISEASE_CLASSES = ("psc", "pbc", "aih")
SEXES = ("F", "M")

# Context feature vector layout used by the models (see dataset.encode_context):
#   [disease_onehot(3), age_norm(1), sex(1), responder(1),
#    udca_active(1), ercp_now(1)]  -> 8 dims, time-varying for treatment flags.
CONTEXT_STATIC_DIM = 6   # disease(3) + age + sex + responder
CONTEXT_TIME_DIM = 2     # udca_active, ercp_now  (depend on t)
CONTEXT_DIM = CONTEXT_STATIC_DIM + CONTEXT_TIME_DIM


@dataclass(frozen=True)
class Constraint:
    """Static description of a field's hard constraint (for eval/audit)."""

    idx: int
    name: str
    lower: float
    upper: float
    monotone: str  # "up", "up_except_ercp", or "free"


CONSTRAINTS = (
    Constraint(F, "F", 0.0, 1.0, "up"),
    Constraint(D, "D", 0.0, 1.0, "up"),
    Constraint(S, "S", 0.0, 1.0, "up_except_ercp"),
    Constraint(P, "P", 0.0, 1.0, "up"),
    Constraint(A, "A", 0.0, 1.0, "free"),
    Constraint(C, "C", 0.0, 1.0, "free"),
    Constraint(M, "M", 0.0, 2.0, "up"),
    Constraint(FLARE, "flare", 0.0, 1.0, "free"),
)


def cirrhosis_stage(fibrosis):
    """Derived cirrhosis stage as a fixed monotone function of F.

    Never stored in the state vector (so it can never disagree with F).
    Returns an integer METAVIR-like stage 0..4. Accepts float or array.
    """
    import numpy as np

    f = np.asarray(fibrosis)
    # Fixed thresholds; monotone non-decreasing in F by construction.
    stage = np.zeros_like(f, dtype=int)
    for thr in (0.2, 0.4, 0.6, 0.8):
        stage = stage + (f >= thr).astype(int)
    return stage if stage.shape else int(stage)
