import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.semilog_irm import SemiLogIRM  # type: ignore
except Exception as e_primary:  # pragma: no cover
    st.error(f"Could not import SemiLogIRM: {e_primary}")
    st.stop()

st.set_page_config(page_title="Semi-Log Monetary Policy", layout="centered")
st.title("📈 Semi-Log Monetary Policy Simulator")

st.markdown(
    """
A concise semi-log policy model used in Curve-style lending views.

**Model**: \(r(u)=r_{min}\cdot (r_{max}/r_{min})^u\), with utilization \(u\in[0,1]\).

- `u = 0` returns `r_min`
- `u = 1` returns `r_max`
- between 0 and 1, rate grows exponentially

Reference: [Curve docs — Semi-Log MP](https://docs.curve.finance/developer/lending/contracts/semilog-mp)
"""
)

st.sidebar.header("Parameters")
preset = st.sidebar.selectbox(
    "Preset profile",
    ["Balanced", "Conservative", "Aggressive"],
    index=0,
    help="Preset values for quick scenario testing.",
)

preset_map = {
    "Balanced": (0.01, 120.0),
    "Conservative": (0.5, 40.0),
    "Aggressive": (0.01, 300.0),
}
def_min_pct, def_max_pct = preset_map[preset]

rate_min_pct = st.sidebar.number_input(
    "Minimum rate r_min (%)",
    min_value=0.001,
    max_value=100.0,
    value=float(def_min_pct),
    step=0.01,
    format="%.3f",
    help="UI is percentage. Internally converted to decimal.",
)
rate_max_pct = st.sidebar.number_input(
    "Maximum rate r_max (%)",
    min_value=max(rate_min_pct + 0.001, 0.01),
    max_value=2000.0,
    value=float(max(def_max_pct, rate_min_pct + 0.5)),
    step=0.1,
    format="%.3f",
    help="UI is percentage. Internally converted to decimal.",
)

utilization_pct = st.slider("Current utilization (%)", 0.0, 100.0, 50.0, 0.1)
log_y = st.sidebar.checkbox("Log-scale Y axis", value=True)

rate_min = rate_min_pct / 100.0
rate_max = rate_max_pct / 100.0
utilization = utilization_pct / 100.0

if utilization < 0.02 or utilization > 0.98:
    st.warning("Utilization is near the boundary (0% or 100%); rates can be very sensitive.")
if rate_max / max(rate_min, 1e-12) > 10_000:
    st.warning("r_max / r_min is very large; curve may appear extremely steep.")

params = {"rate_min": rate_min, "rate_max": rate_max}
if not SemiLogIRM.param_validator(params):
    st.error("Invalid parameters: require positive rates and r_max > r_min.")
    st.stop()

model = SemiLogIRM(rate_min=rate_min, rate_max=rate_max)
current_rate = model.calculate_rate(utilization)
st.metric("Borrow rate (APY %)", f"{current_rate * 100:.2f}%")

st.subheader("Rate curve")
U = np.linspace(0.0, 1.0, 501)
R = np.array([model.calculate_rate(u) for u in U])

fig, ax = plt.subplots()
ax.plot(U * 100, R * 100, linewidth=2, label="Semi-log policy")
ax.axvline(utilization * 100, linestyle=":", linewidth=1, label="current utilization")
ax.axhline(current_rate * 100, linestyle=":", linewidth=1, label="current rate")
if log_y:
    ax.set_yscale("log")
ax.set_xlabel("Utilization (%)")
ax.set_ylabel("Borrow rate (APY %)" )
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

export = {
    "type": "semi_log",
    "params": {
        "rate_min": rate_min,
        "rate_max": rate_max,
    },
}
st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps(export, indent=2),
    file_name="semilog_irm_params.json",
    mime="application/json",
)
