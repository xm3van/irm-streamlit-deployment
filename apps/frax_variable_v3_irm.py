import json
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sys 
import os 

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.frax_variable_irm import FraxVariableRateV3, apr_to_per_second, per_second_to_apr  # type: ignore
except Exception as e_primary:  # pragma: no cover
    st.error(f"Could not import FraxVariableRateV3: {e_primary}")
    st.stop()

st.set_page_config(page_title="Frax Variable V3 Policy Simulator", layout="centered")
st.title("📈 Frax Variable IRM (V3)")

# ---------------- FORMULAS ----------------

st.markdown(
    r"""
### Model

**Full-utilization rate dynamics**

$$
R_{full,new} =
\begin{cases}
R_{old} \cdot \frac{H}{H + (\Delta u)^2 t} & u < u_{min} \\
R_{old} \cdot \frac{H + (\Delta u)^2 t}{H} & u > u_{max} \\
R_{old} & \text{otherwise}
\end{cases}
$$

---

**Borrow rate (piecewise)**

$$
R(u) =
\begin{cases}
R_0 + \frac{u}{u_v}(R_v - R_0) & u < u_v \\
R_v + \frac{u - u_v}{1-u_v}(R_{full} - R_v) & u \ge u_v
\end{cases}
$$

---

Where:
- $H$ = half-life  
- $\Delta u$ = distance from target band  
- $R_v$ = vertex rate  
"""
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("Market Parameters")

min_target_util_pct = st.sidebar.slider("Min target util (%)", 1.0, 99.0, 75.0)
max_target_util_pct = st.sidebar.slider("Max target util (%)", min_target_util_pct + 0.1, 99.5, 85.0)
vertex_util_pct = st.sidebar.slider("Vertex util (%)", 1.0, 99.0, 80.0)

zero_util_rate_pct = st.sidebar.slider("Zero-util APR (%)", 0.0, 6.0, 1.0)
vertex_rate_percent_pct = st.sidebar.slider("Vertex weight (%)", 0.0, 100.0, 20.0)

min_full_util_rate_pct = st.sidebar.slider("Min full-util APR (%)", 0.0, 100.0, 5.0)
max_full_util_rate_pct = st.sidebar.slider("Max full-util APR (%)", min_full_util_rate_pct, 500.0, 100.0)

rate_half_life_hours = st.sidebar.slider("Half-life (hours)", 0.1, 168.0, 12.0)

# ---------------- STATE ----------------

st.sidebar.header("Market State")

utilization_pct = st.sidebar.slider("Utilization (%)", 0.0, 100.0, 80.0)
old_full_util_rate_pct = st.sidebar.slider("Previous full-util APR (%)", 0.0, max_full_util_rate_pct, 20.0)
delta_time_hours = st.sidebar.slider("Elapsed time (hours)", 0.0, 168.0, 12.0)

# ---------------- MODEL ----------------

params = {
    "min_target_util": min_target_util_pct / 100,
    "max_target_util": max_target_util_pct / 100,
    "vertex_utilization": vertex_util_pct / 100,
    "zero_util_rate": apr_to_per_second(zero_util_rate_pct / 100),
    "min_full_util_rate": apr_to_per_second(min_full_util_rate_pct / 100),
    "max_full_util_rate": apr_to_per_second(max_full_util_rate_pct / 100),
    "rate_half_life": rate_half_life_hours * 3600,
    "vertex_rate_percent": vertex_rate_percent_pct / 100,
}

model = FraxVariableRateV3(**params)

utilization = utilization_pct / 100
old_full = apr_to_per_second(old_full_util_rate_pct / 100)
dt = delta_time_hours * 3600

rate, new_full, vertex = model.get_new_rate(dt, utilization, old_full)

# ---------------- METRICS ----------------

c1, c2, c3 = st.columns(3)

c1.metric("Borrow APR", f"{per_second_to_apr(rate)*100:.2f}%")
c2.metric("Full-util APR", f"{per_second_to_apr(new_full)*100:.2f}%")
c3.metric("Vertex APR", f"{per_second_to_apr(vertex)*100:.2f}%")

# ---------------- CURVE ----------------

st.subheader("Rate Curve")

U = np.linspace(0, 1, 400)
R = []

for u in U:
    r, _, _ = model.get_new_rate(dt, u, old_full)
    R.append(per_second_to_apr(r) * 100)

fig, ax = plt.subplots()
ax.plot(U * 100, R, linewidth=2)

ax.axvline(vertex_util_pct, linestyle="--")
ax.axvline(min_target_util_pct, linestyle=":")
ax.axvline(max_target_util_pct, linestyle=":")
ax.axvline(utilization_pct, linestyle="-.")

ax.set_xlabel("Utilization (%)")
ax.set_ylabel("APR %")
ax.grid(True)

st.pyplot(fig)

# ---------------- EXPORT ----------------

st.download_button(
    "Download params",
    json.dumps(params, indent=2),
    file_name="frax_params.json",
)