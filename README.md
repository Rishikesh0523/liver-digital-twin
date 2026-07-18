# Liver Disease World Model — JEPA-style prototype

A predictive world model for an 8-D clinical liver-disease state `x(t)` that is
**accurate**, **constraint-respecting by construction** (one-directional fields
can never reverse), and **explainable**. Built for the Digital Liver take-home.

The primary deliverable is [`DECISION_MEMO.docx`](DECISION_MEMO.docx) — the reasoning
behind every choice. This README is the map and the run instructions.

## Quick start

```bash
pip install -r requirements.txt

# 1) all property + constraint + model tests (fast)
pytest liver_world_model/tests/ -q

# 2) train JEPA + baseline end-to-end (CPU-friendly defaults, ~few min)
python -m liver_world_model.experiments.train  --config liver_world_model/configs/default.yaml

# 3) evaluate: accuracy, constraints, probes, noise robustness, manifold critic,
#    counterfactual faithfulness, worked explanation -> results/evaluation_report.json
python -m liver_world_model.experiments.evaluate --config liver_world_model/configs/default.yaml

# 4) figures (loss curves, collapse, noise, manifold, counterfactual + mermaid diagrams)
python -m liver_world_model.experiments.make_figures --config liver_world_model/configs/default.yaml

# 5) interactive teaching app (generate data, break the constraints, run both models)
streamlit run liver_world_model/app.py
```

Steps 4–6 are optional (need `matplotlib`, `python-docx`, `streamlit`/`plotly`, and — for
rendered mermaid diagrams — the mermaid CLI `mmdc`; each degrades gracefully if missing).

## Interactive app

`streamlit run liver_world_model/app.py` — a six-tab walkthrough built to be explained to
someone seeing the project for the first time:

| Tab | What you can do |
|---|---|
| 📖 **How it works** | the 8-D state, the causal graph, the architecture |
| 🧬 **Make dataset** | turn knobs (disease, susceptibility, UDCA, ERCP) and watch a patient's disease move |
| 🔬 **Explore data** | a whole cohort + a live check that the simulator never breaks the rules |
| 🔒 **Constraints** | **try to break the guarantee** — drag the raw network output to −50 and watch fibrosis refuse to fall; set cholestasis to 0 and watch the cancer hazard freeze |
| 🤖 **Run models** | JEPA vs baseline on the same patient; add sensor noise and toggle the denoised anchor to reproduce the σ≈0.15 crossover live |
| 💡 **Explain** | ablation vs attention for one trajectory — why did it predict that? |

Everything is seeded and device-agnostic (auto-selects CUDA if present). No manual
steps between stages; `evaluate` loads the checkpoints `train` wrote.

## What's here

```
liver_world_model/
├── data/
│   ├── schema.py        # frozen 8-D state contract (indices, bounds, cirrhosis fn)
│   ├── generator.py     # seeded dynamical generator (the "gift and trap")
│   └── dataset.py       # sliding-window torch Dataset, patient-level splits
├── models/
│   ├── constraints.py   # ** by-construction constrained parameterisation **
│   ├── collapse.py      # variance + effective-rank reg; participation-ratio metric
│   ├── encoder.py       # graph-attention encoder (causal-graph attention mask)
│   ├── predictor.py     # GRU latent dynamics (Neural-ODE discussed, deferred)
│   ├── decoder.py       # latent -> valid state (delta-based, ratchet floor)
│   ├── jepa.py          # full JEPA: encode -> predict latent -> decode
│   └── baseline.py      # direct next-state predictor (the peer to beat)
├── training/
│   ├── losses.py        # composite loss + weight schedule
│   ├── objectives.py    # loss closures for JEPA / baseline
│   └── trainer.py       # generic loop: cosine+warmup, clip, early stop, ckpt
├── evaluation/
│   ├── metrics.py       # MSE/MAE per field, constraint & bounds rates, err-vs-horizon
│   ├── probes.py        # 4 generalisation probes + counterfactual note
│   ├── robustness.py    # ** noise-robustness probe (denoised anchor) — where JEPA wins **
│   ├── manifold_critic.py # "0 violations != on-manifold" discriminator
│   ├── counterfactual.py  # UDCA-earlier intervention vs generator re-run
│   └── explainability.py# attention rollout, ablation, latent traj, fused explanation
├── experiments/
│   ├── train.py         # trains both models, saves training history
│   ├── evaluate.py      # honest report -> results/evaluation_report.json
│   ├── make_figures.py  # all plots + rendered mermaid diagrams -> figures/
│   └── make_report.py   # DOCX technical report with citations
├── diagrams/            # mermaid sources (architecture, causal graph, pipeline)
├── figures/             # generated PNGs
├── configs/default.yaml
├── Liver_World_Model_Report.docx   # generated technical report
└── tests/               # test_generator, test_constraints, test_model
```

## Key decisions

1. **JEPA-style predictive latent** over predicting `x(t)` directly: the latent can
   hold an unobserved per-patient *drive* the 8-D state never exposes. We keep it
   honest with a stop-gradient target + collapse regularisation, and pay for it
   with a constrained decoder and attribution tooling.
2. **Constraints by construction, not projection.** The decoder emits *deltas*;
   ratchet fields use `+softplus` increments, so monotonicity is unrepresentable
   to violate. The M hazard is parameterised as `softplus(·)·F·C·dt`, embedding
   the `F·C→M` coupling structurally rather than hoping the loss finds it.
3. **Collapse is a first-class risk.** We measure the latent's participation ratio
   every epoch and regularise variance + effective rank. The metric *earned its
   keep*: it caught collapse (eff-dim fell to ~1.6) driven by the decoder leaning
   on `current` — see memo §2.

## Results summary (real run, default config, CPU)

| Metric | JEPA | Baseline |
|---|---|---|
| Overall MSE (12-step, held-out) | 0.0106 | **0.0074** |
| Constraint-violation rate | **0.000%** | **0.000%** |
| Latent effective dim (of 16) | 8.86 | n/a |

**Note: on the clean single generator the simple baseline beats JEPA on
accuracy** (the latent doesn't help even on rapid progressors — head-to-head 0.0093
baseline vs 0.0134 JEPA). Both are perfect on constraints; the latent is
healthy/un-collapsed.

**But we then measured where the latent *does* win** (Results-B in the memo):

| Extended check | Result |
|---|---|
| Noise robustness (denoised anchor) | JEPA error **flat in noise** (~0.082); overtakes baseline at **σ≈0.15** |
| Manifold critic (AUC 0.97) | both models on-manifold (score ≈ real 0.64), not just constraint-valid |
| Counterfactual vs generator re-run | direction right (**100%** sign agreement), magnitude under-scaled ~1000× (honest) |

The lesson: **a predictive latent buys robustness, not clean-data accuracy** — the
regime a clean single generator can't exercise. Full numbers in
`results/evaluation_report.json`; figures in `figures/`;

## Known limitations

- Single-generator ceiling: "world model vs. generator-inverter" cannot be settled
  from inside one generator — the probes map *where* recovery breaks, not whether
  it's "real biology".
- Rapid progressors (high susceptibility) outside the training mass are
  under-predicted — shown, not hidden, by the susceptibility probe.
- Long-horizon accuracy decays with autoregressive accumulation (constraints hold
  regardless — that decoupling is the point).
- Counterfactuals are out of scope; the memo says what it would take.

## Running tests

```bash
pytest liver_world_model/tests/ -q          # all
pytest liver_world_model/tests/test_constraints.py -q   # the by-construction gate
```
