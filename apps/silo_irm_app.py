import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sys
import os

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
SECONDS_PER_DAY = 24 * 60 * 60

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.silo_irm import SiloInterestRateModelV2  # type: ignore
except Exception as e_primary:
    st.error(f"Could not import SiloInterestRateModelV2: {e_primary}")
    st.stop()


# ----------------- Conversions -----------------

def apr_pct_to_raw_per_second(apr_pct: float) -> float:
    return (apr_pct / 100.0) / SECONDS_PER_YEAR


def per_day_effect_pct_to_raw(effect_pct: float, util_gap: float = 0.10) -> float:
    return (effect_pct / 100.0) / (util_gap * SECONDS_PER_DAY * SECONDS_PER_YEAR)


def beta_per_day_to_raw(beta_per_day: float) -> float:
    return beta_per_day / SECONDS_PER_DAY


# ----------------- App -----------------

st.set_page_config(page_title="Silo IRM V2 Simulator", layout="centered")
st.title("🏦 Silo InterestRateModelV2 Simulator")

st.markdown("""
Stateful utilization-based interest rate model.

**Interpretation**
- Current APR = instantaneous rate (what borrowers pay right now)
- Compound APR eq. = effective rate over the chosen time window
""")

# ----------------- Sidebar -----------------

st.sidebar.header("Market Parameters")
st.sidebar.caption("Governance-controlled configuration of the model.")

ulow_pct = st.sidebar.slider("ulow (%)", 0.0, 100.0, 35.0)
uopt_pct = st.sidebar.slider("uopt (%)", 0.0, 100.0, 70.0)
ucrit_pct = st.sidebar.slider("ucrit (%)", 0.0, 100.0, 90.0)

ki_effect = st.sidebar.number_input("APR drift / day @ +10pp uopt (%)", value=0.03)
kcrit_effect = st.sidebar.number_input("Extra APR / day @ +10pp ucrit (%)", value=0.06)
klow_effect = st.sidebar.number_input("APR reduction / day @ -10pp ulow (%)", value=0.02)
klin_apr = st.sidebar.number_input("Linear APR @ 100% util (%)", value=1.5)
beta_day = st.sidebar.number_input("Tcrit change per day", value=0.01)

# ----------------- Market State -----------------

st.sidebar.header("Market State")
st.sidebar.caption("Path-dependent state of the system.")

utilization_pct = st.sidebar.slider("Current utilization (%)", 0.0, 100.0, 70.0)

regime_time = st.sidebar.slider("Time in regime (hours)", 0.0, 168.0, 0.0)

ri_apr = st.sidebar.number_input("Initial internal APR (%)", value=3.0)
Tcrit_input = st.sidebar.number_input("Stress memory (Tcrit)", value=0.0)

# ----------------- Build raw params -----------------

raw_params = {
    "ulow": ulow_pct / 100.0,
    "uopt": uopt_pct / 100.0,
    "ucrit": ucrit_pct / 100.0,
    "ki": per_day_effect_pct_to_raw(ki_effect),
    "kcrit": per_day_effect_pct_to_raw(kcrit_effect),
    "klow": per_day_effect_pct_to_raw(klow_effect),
    "klin": apr_pct_to_raw_per_second(klin_apr),
    "beta": beta_per_day_to_raw(beta_day),
    "ri": apr_pct_to_raw_per_second(ri_apr),
    "Tcrit": Tcrit_input,
}

# Apply regime evolution
utilization = utilization_pct / 100.0

if regime_time > 0:
    dt = regime_time * 3600

    raw_params["Tcrit"] += raw_params["beta"] * dt

    drift = raw_params["ki"] * (utilization - raw_params["uopt"]) * dt
    raw_params["ri"] = max(0.0, raw_params["ri"] + drift)

# ----------------- Model -----------------

model = SiloInterestRateModelV2(**raw_params)

# ----------------- Evaluation -----------------

time_hours = st.slider("Forward time (hours)", 0.0, 168.0, 1.0)
dt_seconds = time_hours * 3600

current = model.calculate_current_interest_rate(utilization, dt_seconds)
compound = model.calculate_compound_interest_rate(utilization, dt_seconds)

st.metric("Current APR", f"{current['rcur_apr']*100:.2f}%")
st.metric("Compound APR eq.", f"{compound['apr_equivalent']*100:.2f}%")

# ----------------- Curve -----------------

U = np.linspace(0, 1, 200)
R_cur = [model.calculate_current_interest_rate(u, dt_seconds)["rcur_apr"] for u in U]
R_cmp = [model.calculate_compound_interest_rate(u, dt_seconds)["apr_equivalent"] for u in U]

fig, ax = plt.subplots()
ax.plot(U*100, np.array(R_cur)*100, label="Current APR")
ax.plot(U*100, np.array(R_cmp)*100, label="Compound APR")
ax.axvline(utilization_pct, linestyle="--")
ax.set_xlabel("Utilization (%)")
ax.set_ylabel("APR %")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ----------------- Raw params -----------------

with st.expander("Raw parameters"):
    st.json(raw_params)

st.download_button(
    "Download raw params",
    json.dumps(raw_params, indent=2),
    file_name="params.json"
)
