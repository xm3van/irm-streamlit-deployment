import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.piecewise_linear_irm import PieceWiseLinearIRM  # type: ignore
except Exception as e_primary:  # pragma: no cover
    st.error(f"Could not import PieceWiseLinearIRM: {e_primary}")
    st.stop()

st.set_page_config(page_title="Piecewise Linear Monetary Policy", layout="centered")
st.title("📐 Piecewise Linear Monetary Policy Simulator")

st.markdown(
    r"""
### Short description
Two linear segments joined at a target utilization kink \(u_{opt}\).

### Formula
\[
r(u)=
\begin{cases}
  r_0 + r_1u, & u\le u_{opt} \\
  r_0 + r_1u_{opt} + r_2(u-u_{opt}), & u>u_{opt}
\end{cases}
\]

### Explanation
- Left slope (`r1`) governs low-utilization sensitivity.
- Right slope (`r2`) governs post-kink stress response.
- Continuity at the kink avoids rate jumps.

### Intuition
- Higher `r0`: higher baseline rate everywhere.
- Higher `r1`: tighter policy before target utilization.
- Higher `r2`: sharper penalty near full utilization.
"""
)

st.sidebar.header("Parameters")
u_opt_pct = st.sidebar.slider("Kink utilization u_opt (%)", 0.0, 100.0, 80.0, 0.5)
r0_pct = st.sidebar.slider("Base rate r0 (%)", 0.0, 100.0, 2.0, 0.1)
r1_pct = st.sidebar.slider("Slope r1 (pp per 100% util)", 0.0, 300.0, 10.0, 0.5)
r2_pct = st.sidebar.slider("Slope r2 (pp per 100% util)", float(r1_pct), 1500.0, max(35.0, float(r1_pct)), 1.0)
utilization_pct = st.slider("Current utilization (%)", 0.0, 100.0, 50.0, 0.5)
show_construction = st.sidebar.checkbox("Show segment construction", value=True)

params = {
    "r0": r0_pct / 100.0,
    "r1": r1_pct / 100.0,
    "r2": r2_pct / 100.0,
    "u_opt": u_opt_pct / 100.0,
}

valid, msg = PieceWiseLinearIRM.param_validator(params)
if not valid:
    st.error(msg or "Invalid parameters")
    st.stop()

if utilization_pct < 2 or utilization_pct > 98:
    st.warning("Utilization is near 0%/100%; output can be sensitive.")
if params["r2"] > 5.0:
    st.warning("Post-kink slope is very high; rates may rise sharply near full utilization.")

model = PieceWiseLinearIRM(**params)
utilization = utilization_pct / 100.0
current_rate = model.calculate_rate(utilization)
st.metric("Borrow rate (APY %)", f"{current_rate * 100:.2f}%")

# continuity metric around kink
u_opt = params["u_opt"]
eps = 1e-8
left_u = max(0.0, u_opt - eps)
right_u = min(1.0, u_opt + eps)
left_r = model.calculate_rate(left_u)
right_r = model.calculate_rate(right_u)
st.caption(f"Continuity check at kink: |left-right| = {abs(left_r-right_r)*100:.8f} percentage points")

st.subheader("Rate curve")
U = np.linspace(0.0, 1.0, 501)
R = np.array([model.calculate_rate(u) for u in U])

fig, ax = plt.subplots()
ax.plot(U * 100, R * 100, linewidth=2, label="Piecewise policy")
ax.axvline(u_opt * 100, linestyle="--", linewidth=1, label="u_opt")
ax.axvline(utilization * 100, linestyle=":", linewidth=1, label="current utilization")
ax.axhline(current_rate * 100, linestyle=":", linewidth=1, label="current rate")

if show_construction:
    r_at_0 = model.calculate_rate(0.0)
    r_at_1 = model.calculate_rate(1.0)
    ax.plot([0, u_opt * 100], [r_at_0 * 100, left_r * 100], linestyle="--", linewidth=1)
    ax.plot([u_opt * 100, 100], [right_r * 100, r_at_1 * 100], linestyle="--", linewidth=1)

ax.set_xlabel("Utilization (%)")
ax.set_ylabel("Borrow rate (APY %)")
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps({"type": "piecewise_linear", "params": params}, indent=2),
    file_name="piecewise_linear_irm_params.json",
    mime="application/json",
)
