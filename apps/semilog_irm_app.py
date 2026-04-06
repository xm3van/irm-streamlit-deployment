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
    "Semi-log policy maps utilization $u$ to borrow rate $r(u)$ with exponential growth."
)
st.latex(
    r"r(u)=r_{\min}\cdot\left(\frac{r_{\max}}{r_{\min}}\right)^u,\quad u\in[0,1]"
)

st.markdown("Explanation:")
st.markdown(
    """
- At `u = 0`: `r(u) = r_min`
- At `u = 1`: `r(u) = r_max`
- Between 0 and 1: growth is smooth but increasingly steep.
"""
)

st.markdown("Intuition:")
st.markdown(
    """
- `r_min`: floor rate when liquidity is abundant.
- `r_max`: stress rate near full utilization.
- Larger `r_max / r_min`: sharper upward curve as utilization rises.
"""
)

st.markdown(
    "Reference: [Curve docs — Semi-Log MP](https://docs.curve.finance/developer/lending/contracts/semilog-mp)"
)

st.sidebar.header("Parameters")
rate_min_pct = st.sidebar.slider(
    "Minimum rate r_min (%)",
    min_value=0.001,
    max_value=50.0,
    value=0.010,
    step=0.001,
)

rate_max_pct = st.sidebar.slider(
    "Maximum rate r_max (%)",
    min_value=max(rate_min_pct + 0.001, 0.01),
    max_value=500.0,
    value=max(50.0, rate_min_pct + 1.0),
    step=0.5,
)

utilization_pct = st.slider("Current utilization (%)", 0.0, 100.0, 50.0, 0.1)

rate_min = rate_min_pct / 100.0
rate_max = rate_max_pct / 100.0
utilization = utilization_pct / 100.0

if utilization < 0.02 or utilization > 0.98:
    st.warning("Utilization is near 0%/100%; rates become highly sensitive.")
if rate_max / max(rate_min, 1e-12) > 10_000:
    st.warning("Very high r_max / r_min selected; curve can look near-vertical.")

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
ax.set_xlabel("Utilization (%)")
ax.set_ylabel("Borrow rate (APY %)")
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps({"type": "semi_log", "params": params}, indent=2),
    file_name="semilog_irm_params.json",
    mime="application/json",
)
