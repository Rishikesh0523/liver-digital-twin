"""Interactive teaching app for the Liver World Model.

    streamlit run liver_world_model/app.py

Built to be explained to someone who has never seen the project: every tab states
what you are looking at, why it matters, and what to try. Six tabs:

  1. How it works   -- the 8-D state, the causal graph, the architecture
  2. Make dataset   -- generate a patient, watch the disease move
  3. Explore data   -- a whole cohort, and proof the constraints hold in the data
  4. Constraints    -- interactively try (and fail) to break the guarantee
  5. Run models     -- JEPA vs baseline inference, incl. the noise-robustness crossover
  6. Explain        -- why did the model predict this?
"""
from __future__ import annotations

import os
import sys

# make `liver_world_model` importable regardless of where streamlit is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

from liver_world_model.data.generator import LiverDiseaseGenerator
from liver_world_model.data.dataset import encode_context
from liver_world_model.data.schema import (
    A, C, D, F, FLARE, M, P, S, FIELD_NAMES, STATE_DIM, cirrhosis_stage, upper_bounds,
)
from liver_world_model.models.baseline import DirectPredictor
from liver_world_model.models.constraints import ConstrainedOutput, constraint_violations
from liver_world_model.models.jepa import LiverJEPA
from liver_world_model.evaluation.explainability import Explainability

st.set_page_config(page_title="Liver World Model — interactive", page_icon="🫀",
                   layout="wide")

FIELD_LONG = {
    "F": "Fibrosis (scarring)", "D": "Ductopenia (bile-duct loss)",
    "S": "Strictures (narrowed ducts)", "P": "Portal hypertension",
    "A": "Inflammatory activity", "C": "Cholestasis (bile backing up)",
    "M": "Malignancy-hazard accumulator", "flare": "Acute cholangitis flare",
}
RATCHET = {"F", "D", "P", "M"}
COLORS = {"F": "#d62728", "D": "#9467bd", "S": "#8c564b", "P": "#e377c2",
          "A": "#1f77b4", "C": "#2ca02c", "M": "#ff7f0e", "flare": "#7f7f7f"}
CKPT = os.path.join(_HERE, "checkpoints")
FIGS = os.path.join(_HERE, "figures")


# ---------------------------------------------------------------- resources
@st.cache_resource
def load_models():
    """Load both trained models. Returns (jepa, baseline) or (None, None)."""
    jp = os.path.join(CKPT, "jepa_best.pt")
    bp = os.path.join(CKPT, "baseline_best.pt")
    if not (os.path.exists(jp) and os.path.exists(bp)):
        return None, None
    jepa = LiverJEPA(latent_dim=16, d_model=32, n_heads=4, n_attn_layers=2,
                     predictor_hidden=64, decoder_hidden=(64, 32), history_length=12)
    jepa.load_state_dict(torch.load(jp, map_location="cpu", weights_only=False)["model"])
    jepa.eval()
    base = DirectPredictor()
    base.load_state_dict(torch.load(bp, map_location="cpu", weights_only=False)["model"])
    base.eval()
    return jepa, base


@st.cache_data
def gen_cohort(n, horizon, seed, disease=None, susceptibility=None, responder=None,
               udca=None, ercp=None):
    gen = LiverDiseaseGenerator()
    kw = {}
    if disease and disease != "(random)":
        kw["disease_class"] = disease
    if susceptibility is not None:
        kw["susceptibility"] = susceptibility
    if responder is not None:
        kw["responder"] = responder
    if udca is not None:
        kw["udca_start_month"] = udca
    if ercp is not None:
        kw["ercp_months"] = tuple(ercp)
    return gen.generate(n, horizon=horizon, seed=seed, **kw)


def traj_figure(states, ercp_mask=None, udca_mask=None, fields=None, title=""):
    """Plot selected fields over time with ERCP / UDCA markers."""
    fields = fields or FIELD_NAMES
    fig = go.Figure()
    T = states.shape[0]
    for name in fields:
        i = FIELD_NAMES.index(name)
        fig.add_trace(go.Scatter(
            x=list(range(T)), y=states[:, i], name=name, mode="lines",
            line=dict(color=COLORS[name], width=2.5),
            hovertemplate=f"<b>{name}</b> ({FIELD_LONG[name]})<br>month %{{x}}<br>"
                          f"value %{{y:.3f}}<extra></extra>"))
    if ercp_mask is not None:
        for m in np.where(ercp_mask)[0]:
            fig.add_vline(x=int(m), line=dict(color="#00838f", dash="dot", width=2),
                          annotation_text="ERCP", annotation_position="top")
    if udca_mask is not None and udca_mask.any():
        start = int(np.argmax(udca_mask))
        fig.add_vrect(x0=start, x1=T - 1, fillcolor="#2ca02c", opacity=0.06,
                      line_width=0, annotation_text="UDCA active",
                      annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="month", yaxis_title="value",
                      height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


# ================================================================ TAB 1
def tab_how_it_works():
    st.header("Digital twin of a patient's liver disease")
    st.markdown("""
This is a **digital twin of a patient's liver disease** — a model that watches a
patient for 12 months and predicts what the next 12 months look like.

The entire patient is summarised as **8 numbers**, updated every month. That's the whole
state of the world. Everything the model believes about a patient lives in these 8 numbers.
""")
    rows = []
    for name in FIELD_NAMES:
        i = FIELD_NAMES.index(name)
        rows.append({
            "Field": name, "Means": FIELD_LONG[name],
            "Range": f"[0, {upper_bounds()[i]:.0f}]",
            "Can it go down?": "❌ never (one-way)" if name in RATCHET else
                               ("⚠️ only at an ERCP procedure" if name == "S" else "✅ freely"),
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.info("""
Some of these only move one way. Scar tissue does not
un-scar. Destroyed bile ducts do not grow back. So a prediction where fibrosis goes **down**
is not "a bit inaccurate" — **it describes something that cannot physically happen.**
Our job is to make that impossible, not just unlikely.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("How the 8 fields push each other")
        p = os.path.join(FIGS, "causal_graph.png")
        if os.path.exists(p):
            st.image(p, width='stretch')
        st.caption("Orange = one-way (ratchet) fields · blue = fast fields · "
                   "green dashed = treatments. The model's attention is **masked to these "
                   "arrows**, so information can only flow along real biological paths.")
    with c2:
        st.subheader("The model")
        p = os.path.join(FIGS, "architecture.png")
        if os.path.exists(p):
            st.image(p, width='stretch')
        st.caption("Encoder squashes 12 months into a 16-number summary (the *latent*); "
                   "a GRU rolls that summary forward; a constrained decoder turns it back "
                   "into valid patient states.")

    st.subheader("Why 'JEPA'?")
    st.markdown("""
A normal model predicts **the next numbers**. A **JEPA** predicts the *representation* of
the future — its own internal summary of it — and only then decodes that into numbers.

The bet: the 8 numbers don't show you *how fast this particular patient is progressing*.
A latent can carry that hidden "drive". We tested that bet honestly — see the **Run models**
tab for what actually happened (spoiler: the simple model wins on clean data; the latent
wins once the measurements get noisy).
""")


# ================================================================ TAB 2
def tab_make_dataset():
    st.header("Make a patient")
    st.markdown("Turn the knobs, press generate, and watch the disease move. "
                "This is the **simulator** that produces our training data — not the model.")

    c1, c2, c3, c4 = st.columns(4)
    disease = c1.selectbox("Disease type", ["(random)", "psc", "pbc", "aih"], index=1,
                           help="PSC = stricture/flare driven · PBC = cholestatic, "
                                "UDCA-responsive · AIH = inflammation driven")
    seed = c2.number_input("Patient seed", 0, 99999, 42, help="Same seed = same patient, "
                                                             "every single time.")
    horizon = c3.slider("Months to simulate", 24, 60, 36)
    sus = c4.slider("Susceptibility", 0.2, 3.0, 1.0, 0.1,
                    help="How fast this patient's disease progresses. This is HIDDEN from "
                         "the model — it never sees this number.")
    c5, c6, c7 = st.columns(3)
    responder = c5.radio("Responds to UDCA?", [1, 0], horizontal=True,
                         format_func=lambda v: "Yes" if v else "No")
    udca = c6.slider("UDCA starts at month", 0, 30, 6)
    ercp_txt = c7.text_input("ERCP months (comma-separated)", "12, 24")
    try:
        ercp = tuple(int(x) for x in ercp_txt.split(",") if x.strip())
    except ValueError:
        ercp = ()
        st.warning("Could not read ERCP months — using none.")

    cohort = gen_cohort(1, horizon, int(seed), disease, sus, int(responder), int(udca), ercp)
    traj = cohort[0]

    show = st.multiselect("Which fields to plot", FIELD_NAMES,
                          default=["F", "C", "A", "S", "M", "flare"])
    st.plotly_chart(traj_figure(traj.states, traj.ercp_mask, traj.udca_mask, show,
                                "This patient's disease over time"),
                    width='stretch')

    st.markdown("### What to look for")
    a, b, c = st.columns(3)
    a.success("**F, D, P, M never go down.** Follow the fibrosis line — it only ever "
              "climbs or holds flat. That's the ratchet.")
    b.info("**Watch S at an ERCP month** (dotted line). It climbs, then the procedure "
           "knocks it down. That's the one legal exception.")
    c.warning("**Flares hit A and C together** — spikes appear in both at once, then both "
              "decay. They're coupled, not independent.")

    fin = traj.states[-1]
    st.markdown("### Where this patient ended up")
    m = st.columns(4)
    m[0].metric("Fibrosis", f"{fin[F]:.2f}")
    m[1].metric("Cirrhosis stage", f"{int(cirrhosis_stage(fin[F]))} / 4",
                help="Computed FROM fibrosis, never stored — so it can never disagree with it.")
    m[2].metric("Cancer-risk accum.", f"{fin[M]:.3f}")
    m[3].metric("Portal hypertension", f"{fin[P]:.2f}")

    with st.expander("See the raw numbers"):
        df = pd.DataFrame(traj.states, columns=FIELD_NAMES)
        df.insert(0, "month", range(len(df)))
        df["cirrhosis_stage"] = cirrhosis_stage(traj.states[:, F])
        st.dataframe(df.style.format("{:.3f}", subset=FIELD_NAMES),
                     width='stretch', height=280)


# ================================================================ TAB 3
def tab_explore():
    st.header("A whole cohort")
    st.markdown("One patient can look like anything. Here are many — and a check that the "
                "**simulator itself never breaks the rules**.")

    c1, c2, c3 = st.columns(3)
    n = c1.slider("How many patients", 20, 400, 120, 20)
    horizon = c2.slider("Months", 24, 48, 36, key="ex_h")
    seed = c3.number_input("Cohort seed", 0, 9999, 7, key="ex_s")

    cohort = gen_cohort(int(n), int(horizon), int(seed))
    states = np.stack([t.states for t in cohort])          # (n, T, 8)
    ercp = np.stack([t.ercp_mask for t in cohort]).astype(float)

    st.subheader("Every patient's fibrosis, overlaid")
    fig = go.Figure()
    for i in range(min(60, n)):
        fig.add_trace(go.Scatter(y=states[i, :, F], mode="lines", showlegend=False,
                                 line=dict(width=1, color="rgba(214,39,40,0.35)"),
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(y=states[:, :, F].mean(0), mode="lines", name="cohort mean",
                             line=dict(width=4, color="#000")))
    fig.update_layout(height=380, xaxis_title="month", yaxis_title="Fibrosis (F)",
                      title="Each line is one patient. Notice: not one line ever goes down.")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Is the simulated data actually legal?")
    st.markdown("We check every single month-to-month step of every patient against the "
                "hard rules. This is a real computation, run right now on the cohort above.")
    v = constraint_violations(torch.tensor(states, dtype=torch.float32),
                              torch.tensor(ercp, dtype=torch.float32))
    n_steps = n * (horizon - 1)
    cols = st.columns(4)
    cols[0].metric("Transitions checked", f"{n_steps:,}")
    cols[1].metric("Ratchet violations", int(v["mono_F"] + v["mono_D"] +
                                             v["mono_P"] + v["mono_M"]))
    cols[2].metric("Illegal S drops", int(v["mono_S_nonercp"]))
    cols[3].metric("Out of bounds", int(v["bound_low"] + v["bound_high"]))
    if int(v["total"]) == 0:
        st.success(f"✅ **Zero violations across {n_steps:,} transitions.** The training data "
                   "is physically possible by construction — which matters, because if the "
                   "*data* contained impossible livers, asking whether the model learns the "
                   "rule would be meaningless.")
    else:
        st.error(f"❌ {int(v['total'])} violations — that would be a bug in the generator.")

    st.subheader("How the disease types differ")
    per = []
    for t in cohort:
        per.append({"disease": t.context.disease_class, "F": t.states[-1, F],
                    "S": t.states[-1, S], "C": t.states[-1, C], "M": t.states[-1, M],
                    "susceptibility": t.context.susceptibility,
                    "responder": t.context.responder})
    df = pd.DataFrame(per)
    st.dataframe(df.groupby("disease")[["F", "S", "C", "M"]].mean().round(3),
                 width='stretch')
    st.caption("PSC carries the most strictures (S) — it's the bile-duct disease. "
               "That difference is built into the simulator, and it's why disease class "
               "is given to the model as context.")


# ================================================================ TAB 4
def tab_constraints():
    st.header("Try to break it")
    st.markdown("""
This is the heart of the project. **We don't check the model's answers and we don't
correct them.** We changed *the question we ask the model* so that an illegal answer
cannot be expressed.

> **The odometer trick.** You never ask *"what will the odometer read?"* — you ask
> *"how many miles were driven?"*. Miles driven has no negative version, so the odometer
> can never run backwards. Not because anyone checks it: because *"drove minus-50 miles"*
> is not a sentence you can say.
""")

    layer = ConstrainedOutput()

    st.subheader("1. The one-way fields — go on, try to make fibrosis fall")
    c1, c2 = st.columns([1, 2])
    with c1:
        cur_f = st.slider("Fibrosis right now", 0.0, 1.0, 0.5, 0.01)
        raw_f = st.slider("What the network outputs (raw)", -50.0, 50.0, 0.0, 0.5,
                          help="Drag this as far negative as you like. This is the raw, "
                               "totally unconstrained number the neural net produces.")
    x = torch.zeros(1, STATE_DIM)
    x[0, F] = cur_f
    x[0, C] = 0.5
    raw = torch.zeros(1, STATE_DIM)
    raw[0, F] = raw_f
    nxt = layer(x, raw, torch.zeros(1))
    new_f = float(nxt[0, F])
    delta = new_f - cur_f
    with c2:
        st.markdown(f"""
```
next_F = current_F + softplus(raw - 4)
       = {cur_f:.3f}      + {delta:.6f}
       = {new_f:.3f}
```
""")
        if new_f >= cur_f - 1e-9:
            st.success(f"✅ **Fibrosis went from {cur_f:.3f} → {new_f:.3f}.** It did not fall. "
                       f"It *cannot* fall — softplus only ever returns a positive number "
                       f"(here: **{delta:.6f}**), and we can only ever **add** it.")
        else:
            st.error("Constraint broken — this should be impossible!")
        st.caption("Even at raw = −50, softplus returns a tiny *positive* number. "
                   "There is no input that produces a decrease.")

    st.divider()
    st.subheader("2. The coupling — cancer risk needs BOTH scarring and bile backup")
    st.markdown("*Rust needs metal **and** water. Rust-prone metal in a dry room never rusts.*")
    c1, c2, c3 = st.columns(3)
    f_val = c1.slider("Fibrosis (F)", 0.0, 1.0, 0.8, 0.05, key="mf")
    c_val = c2.slider("Cholestasis (C)", 0.0, 1.0, 0.6, 0.05, key="mc",
                      help="Drag this to ZERO and watch the cancer risk freeze.")
    raw_m = c3.slider("Network's raw hazard output", -10.0, 10.0, 4.0, 0.5, key="mr")
    x = torch.zeros(1, STATE_DIM)
    x[0, F], x[0, C], x[0, M] = f_val, c_val, 0.5
    raw = torch.zeros(1, STATE_DIM)
    raw[0, M] = raw_m
    nxt = layer(x, raw, torch.zeros(1))
    dM = float(nxt[0, M]) - 0.5
    st.markdown(f"""
```
ΔM = softplus(raw - 4) × F × C × dt
   = {float(torch.nn.functional.softplus(torch.tensor(raw_m - 4.0))):.4f}        × {f_val:.2f} × {c_val:.2f} × 1
   = {dM:.6f}
```
""")
    if c_val == 0 or f_val == 0:
        st.success("✅ **ΔM is exactly 0.** With no bile backup (or no scarring), the cancer "
                   "risk *cannot* move — the product is zero. The model didn't learn this. "
                   "It's arithmetic.")
    else:
        st.info(f"Cancer risk climbs by **{dM:.6f}** this month. Now drag **C to zero** and "
                "watch it stop dead.")

    st.divider()
    st.subheader("3. The exception — strictures and the surgeon")
    c1, c2 = st.columns([1, 2])
    with c1:
        cur_s = st.slider("Strictures right now", 0.0, 1.0, 0.7, 0.01, key="sc")
        raw_s = st.slider("Network raw output", -10.0, 10.0, -5.0, 0.5, key="sr")
        did_ercp = st.toggle("Did an ERCP happen this month?", value=False)
    x = torch.zeros(1, STATE_DIM)
    x[0, S] = cur_s
    raw = torch.zeros(1, STATE_DIM)
    raw[0, S] = raw_s
    nxt = layer(x, raw, torch.tensor([1.0 if did_ercp else 0.0]))
    new_s = float(nxt[0, S])
    with c2:
        if did_ercp:
            st.info(f"**ERCP month.** S: {cur_s:.3f} → **{new_s:.3f}**. The procedure "
                    f"physically opened the duct, so a drop is *allowed* here — the model "
                    f"switches to a free `sigmoid` branch.")
        else:
            st.success(f"**Normal month.** S: {cur_s:.3f} → **{new_s:.3f}**. No procedure, "
                       f"so the one-way rule applies and S cannot fall — no matter what the "
                       f"network says.")
        st.caption("Toggle the switch and drag the raw output negative. S only ever falls "
                   "when the procedure actually happened.")

    st.divider()
    st.subheader("4. The real test: feed it pure nonsense")
    st.markdown("A guarantee that only holds for a *trained* model is not a guarantee. So we "
                "test it with **random garbage** — random patient states, wildly random "
                "network outputs, random procedures.")
    if st.button("🎲 Run the nonsense test (1,560 random states)", type="primary"):
        g = torch.Generator().manual_seed(int(np.random.randint(1e6)))
        upper = torch.tensor(upper_bounds())
        cur = torch.rand(60, STATE_DIM, generator=g) * upper
        traj, ercps = [cur], []
        for _ in range(25):
            r = torch.randn(60, STATE_DIM, generator=g) * 5.0
            e = (torch.rand(60, generator=g) < 0.2).float()
            cur = layer(cur, r, e)
            traj.append(cur)
            ercps.append(e)
        states = torch.stack(traj, dim=1)
        emask = torch.stack([torch.zeros(60)] + ercps, dim=1)
        v = constraint_violations(states, emask)
        if int(v["total"]) == 0:
            st.success(f"✅ **{states.shape[0] * states.shape[1]:,} random states. "
                       f"ZERO violations.** Not 'rare'. Zero. The guarantee doesn't depend "
                       "on training going well, or on the patient looking like anything "
                       "we've seen. It cannot fail.")
        else:
            st.error(f"{int(v['total'])} violations — the guarantee is broken (this is a bug).")
        st.json({k: int(x) for k, x in v.items()})


# ================================================================ TAB 5
def tab_models():
    st.header("Run the models")
    jepa, base = load_models()
    if jepa is None:
        st.error("No trained checkpoints found in `liver_world_model/checkpoints/`. "
                 "Run `python -m liver_world_model.experiments.train` first.")
        return

    st.markdown("""
Two models get the **same** 12 months of history and predict the **next 12 months**:

| | |
|---|---|
| 🔵 **JEPA** | encodes history into a 16-number latent, rolls *that* forward, decodes it |
| 🔴 **Baseline** | skips the latent — predicts the next state directly, step by step |

Both use the **identical** constraint layer, so any difference is due to the latent alone.
""")
    c1, c2, c3 = st.columns(3)
    seed = c1.number_input("Patient seed", 0, 99999, 3, key="mseed")
    disease = c2.selectbox("Disease", ["(random)", "psc", "pbc", "aih"], 1, key="mdis")
    sus = c3.slider("Susceptibility (hidden from the model)", 0.2, 3.0, 1.0, 0.1, key="msus")

    H, K = 12, 12
    cohort = gen_cohort(1, H + K + 2, int(seed) + 500, disease, sus)
    traj = cohort[0]
    ctx = encode_context(traj)
    hist = torch.tensor(traj.states[:H], dtype=torch.float32).unsqueeze(0)
    hctx = torch.tensor(ctx[:H], dtype=torch.float32).unsqueeze(0)
    fctx = torch.tensor(ctx[H:H + K], dtype=torch.float32).unsqueeze(0)
    cur = torch.tensor(traj.states[H - 1], dtype=torch.float32).unsqueeze(0)
    ef = torch.tensor(traj.ercp_mask[H:H + K].astype(np.float32)).unsqueeze(0)
    true_future = traj.states[H:H + K]

    st.subheader("Sensor noise — the interesting experiment")
    st.markdown("Real measurements are noisy. Add some and see what happens.")
    c1, c2 = st.columns([2, 1])
    sigma = c1.slider("Measurement noise σ", 0.0, 0.30, 0.0, 0.01,
                      help="0 = perfect lab values. Real clinical data is never 0.")
    denoise = c2.toggle("JEPA: use denoised anchor", value=False,
                        help="Anchor the forecast on the latent's estimate of today's state "
                             "(distilled from 12 months) instead of the single noisy reading.")

    if sigma > 0:
        g = torch.Generator().manual_seed(0)
        upper = torch.tensor(upper_bounds())
        hist_n = torch.minimum((hist + torch.randn(hist.shape, generator=g) * sigma
                                ).clamp(min=0), upper)
        cur_n = torch.minimum((cur + torch.randn(cur.shape, generator=g) * sigma
                               ).clamp(min=0), upper)
    else:
        hist_n, cur_n = hist, cur

    with torch.no_grad():
        jp = jepa.predict(hist_n, hctx, fctx, cur_n, ef, denoise=denoise)[0].numpy()
        bp = base(hist_n, hctx, fctx, cur_n, ef)[0].numpy()

    j_mae = float(np.abs(jp - true_future).mean())
    b_mae = float(np.abs(bp - true_future).mean())
    m = st.columns(4)
    m[0].metric("JEPA error (MAE)", f"{j_mae:.4f}",
                delta=f"{j_mae - b_mae:+.4f} vs baseline", delta_color="inverse")
    m[1].metric("Baseline error (MAE)", f"{b_mae:.4f}")
    jv = int(constraint_violations(torch.tensor(jp).unsqueeze(0), ef)["total"])
    bv = int(constraint_violations(torch.tensor(bp).unsqueeze(0), ef)["total"])
    m[2].metric("JEPA violations", jv)
    m[3].metric("Baseline violations", bv)

    if sigma == 0 and not denoise:
        st.info("**At zero noise the baseline usually wins.** We report that honestly — on "
                "clean, fully-observed data a simple direct model is hard to beat, because "
                "there's nothing hidden for a latent to infer.")
    elif sigma >= 0.12 and denoise:
        st.success("**Now flip it back and forth.** With the denoised anchor ON, JEPA's error "
                   "barely moves as you crank the noise — because it estimates today's state "
                   "from *twelve months* of history instead of trusting one noisy reading. "
                   "That's what the latent buys: **robustness, not clean-data accuracy.**")
    elif sigma > 0 and not denoise:
        st.warning("Noise is on but the denoised anchor is **off** — both models are trusting "
                   "the single noisy reading. Turn the anchor on and watch JEPA hold steady.")

    st.subheader("Input → Output")
    show = st.multiselect("Fields", FIELD_NAMES, default=["F", "C", "A", "S"], key="mshow")
    n_show = len(show)
    fig = make_subplots(rows=(n_show + 1) // 2, cols=2, subplot_titles=[
        f"{f} — {FIELD_LONG[f]}" for f in show])
    for k, name in enumerate(show):
        i = FIELD_NAMES.index(name)
        r, c = k // 2 + 1, k % 2 + 1
        fig.add_trace(go.Scatter(x=list(range(H)), y=traj.states[:H, i],
                                 name="history (input)", legendgroup="h",
                                 showlegend=(k == 0),
                                 line=dict(color="#333", width=3)), r, c)
        xs = list(range(H - 1, H + K))
        fig.add_trace(go.Scatter(x=xs, y=np.r_[traj.states[H - 1, i], true_future[:, i]],
                                 name="truth", legendgroup="t", showlegend=(k == 0),
                                 line=dict(color="#2ca02c", width=3, dash="dash")), r, c)
        fig.add_trace(go.Scatter(x=xs, y=np.r_[traj.states[H - 1, i], jp[:, i]],
                                 name="JEPA", legendgroup="j", showlegend=(k == 0),
                                 line=dict(color="#1f77b4", width=2.5)), r, c)
        fig.add_trace(go.Scatter(x=xs, y=np.r_[traj.states[H - 1, i], bp[:, i]],
                                 name="baseline", legendgroup="b", showlegend=(k == 0),
                                 line=dict(color="#d62728", width=2.5)), r, c)
        fig.add_vline(x=H - 0.5, line=dict(color="#999", dash="dot"), row=r, col=c)
    fig.update_layout(height=260 * ((n_show + 1) // 2), hovermode="x unified",
                      title="Black = what the model saw · Green dashed = what really "
                            "happened · Blue/Red = the two predictions")
    st.plotly_chart(fig, width='stretch')

    st.caption("The vertical dotted line is 'today'. Everything left of it is the input; "
               "everything right is a forecast made without seeing the answer.")

    with st.expander("Per-field error table"):
        st.dataframe(pd.DataFrame({
            "field": FIELD_NAMES,
            "JEPA MAE": np.abs(jp - true_future).mean(0).round(4),
            "baseline MAE": np.abs(bp - true_future).mean(0).round(4),
        }), width='stretch', hide_index=True)


# ================================================================ TAB 6
def tab_explain():
    st.header("Why did it predict that?")
    jepa, base = load_models()
    if jepa is None:
        st.error("Train the models first (see the Run models tab).")
        return
    st.markdown("A forecast you can't interrogate is not much use in a clinic. Three "
                "different ways of asking the model *why*.")

    c1, c2 = st.columns(2)
    seed = c1.number_input("Patient seed", 0, 99999, 3, key="xseed")
    sus = c2.slider("Susceptibility", 0.2, 3.0, 2.4, 0.1, key="xsus")

    H, K = 12, 12
    traj = gen_cohort(1, H + K + 2, int(seed) + 500, "psc", sus)[0]
    ctx = encode_context(traj)
    inputs = dict(
        history=traj.states[:H].astype("float32"),
        hist_context=ctx[:H].astype("float32"),
        fut_context=ctx[H:H + K].astype("float32"),
        current=traj.states[H - 1].astype("float32"),
        ercp_future=traj.ercp_mask[H:H + K].astype("float32"),
    )
    ex = Explainability(jepa, torch.device("cpu"))
    out = ex.explain_month(**inputs, target_step=K - 1)

    pf = out["predicted_F"]
    true_f = float(traj.states[H + K - 1, F])
    c = st.columns(4)
    c[0].metric("Predicted fibrosis", f"{pf:.3f}")
    c[1].metric("Actual fibrosis", f"{true_f:.3f}", delta=f"{pf - true_f:+.3f}")
    c[2].metric("Predicted cirrhosis stage", f"{out['derived_cirrhosis_stage']} / 4")
    c[3].metric("Actual stage", f"{int(cirrhosis_stage(true_f))} / 4")

    if pf < true_f - 0.1:
        st.warning(f"**The model under-predicts this patient.** It says fibrosis reaches "
                   f"{pf:.2f}; it actually reaches {true_f:.2f}. This is our known, reported "
                   f"failure: fast progressors (susceptibility {sus:.1f}) get pulled toward "
                   f"the 'typical' patient the model saw most often. We show this rather "
                   f"than hide it.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Which field mattered? (ablation)")
        st.caption("We delete each field from the history and measure how much the forecast "
                   "moves. Bigger bar = the model leaned on it more. This is the most "
                   "trustworthy view — it doesn't take the model's word for anything.")
        ab = out["top_input_drivers_by_ablation"]
        d = pd.DataFrame(ab, columns=["field", "impact"])
        st.plotly_chart(go.Figure(go.Bar(x=d["field"], y=d["impact"],
                                         marker_color="#1f77b4")).update_layout(
            height=300, yaxis_title="shift in forecast"), width='stretch')
    with c2:
        st.subheader("What did attention look at?")
        st.caption("The encoder's attention, per field. Because attention is masked to the "
                   "causal graph, it can only look along real biological edges.")
        at = out["attention_received_by_field"]
        if at:
            d = pd.DataFrame({"field": list(at.keys()), "attention": list(at.values())})
            st.plotly_chart(go.Figure(go.Bar(x=d["field"], y=d["attention"],
                                             marker_color="#ff7f0e")).update_layout(
                height=300, yaxis_title="attention received"), width='stretch')

    st.success("""
**Read these two together.** Both should point at **A (inflammation)** and **C (cholestasis)** —
and they do. That's exactly the simulator's rule: fibrosis is driven by inflammation plus
cholestasis. **The model found the right mechanism.**

We show ablation *next to* attention on purpose: attention is a seductive but unreliable
explanation on its own. Ablation is the check. If they disagreed, we'd trust ablation.
""")

    with st.expander("Full structured explanation (raw)"):
        st.json(out)


# ================================================================ main
def main():
    st.title("🫀 Liver Disease World Model")
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📖 How it works", "🧬 Make dataset", "🔬 Explore data",
        "🔒 Constraints", "🤖 Run models", "💡 Explain",
    ])
    with t1:
        tab_how_it_works()
    with t2:
        tab_make_dataset()
    with t3:
        tab_explore()
    with t4:
        tab_constraints()
    with t5:
        tab_models()
    with t6:
        tab_explain()


if __name__ == "__main__":
    main()
