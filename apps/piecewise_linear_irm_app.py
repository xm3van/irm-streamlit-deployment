# Piecewise Linear IRM — Streamlit App
# Save as: piecewise_linear_irm_app.py
# Run with:  python -m streamlit run piecewise_linear_irm_app.py

import json
import os
import sys
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ── Import PieceWiseLinearIRM from your project ─────────────────────────────
# Try the path you requested first, then fallback to older name.
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.piecewise_linear_irm import PieceWiseLinearIRM  # type: ignore
except Exception as e_primary:  # pragma: no cover
    try:
        from interest_rate_models.piecewise_linear import PieceWiseLinearIRM  # type: ignore
    except Exception as e_fallback:
        st.error(
            "Could not import PieceWiseLinearIRM from either:\n\n"
            " • interest_rate_models.piecewise_linear_irm\n"
            " • interest_rate_models.piecewise_linear\n\n"
            f"Primary error: {e_primary}\nFallback error: {e_fallback}"
        )
        st.stop()

# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Piecewise Linear IRM", layout="centered")
st.title("📐 Piece-wise Linear IRM Simulator")

st.markdown(
    r"""
This app lets you explore a **piece-wise linear** interest-rate model (IRM). It is continuous at the
**target utilization** $u_{opt}$ and allows different slopes before/after the kink.

**Formula**  
$$
r(u)=
\begin{cases}
  r_0 + r_1 u, & u \le u_{opt} \\
  r_0 + r_1 u_{opt} + r_2 (u-u_{opt}), & u > u_{opt}
\end{cases}
$$
"""
)

# Sidebar — Parameters (all sliders, as requested)
st.sidebar.header("Parameters")

u_opt = st.sidebar.slider(
    "u_opt (kink)", min_value=0.0, max_value=1.0, value=0.80, step=0.01,
    help="Target utilization where the slope can change."
)

r0 = st.sidebar.slider(
    "r0 (base)", min_value=0.0, max_value=0.5, value=0.02, step=0.005,
    help="Base intercept rate at u = 0."
)

r1 = st.sidebar.slider(
    "r1 (slope ≤ u_opt)", min_value=0.0, max_value=1.0, value=0.10, step=0.01,
    help="Slope for the left segment (encourages borrowing at low u)."
)

# Ensure r2 ≥ r1 by constraining min dynamically
_default_r2 = float(max(0.01, r1))
r2 = st.sidebar.slider(
    "r2 (slope > u_opt)", min_value=float(r1), max_value=50.0,
    value=_default_r2, step=0.01,
    help="Slope for the right segment. Typically r2 ≥ r1 for sharper response."
)

show_construction = st.sidebar.checkbox(
    "Show segment construction", value=True,
    help="Draw dashed helper lines to visualize kink continuity."
)

# Main — utilization as a slider (requested)
utilization = st.slider(
    "📌 Current Utilization", 0.0, 1.0, 0.50, step=0.01,
    help="Set the current market utilization to evaluate r(u)."
)

# ── Validate + instantiate model ────────────────────────────────────────────
params = {"r0": r0, "r1": r1, "r2": r2, "u_opt": u_opt}

# Accept validators that return bool OR (bool, message)
try:
    _pv_res = PieceWiseLinearIRM.param_validator(params)
except Exception as _pv_exc:  # If validator itself errors, show a helpful message
    st.error(f"Parameter validation raised an exception: {_pv_exc}")
    st.stop()

if isinstance(_pv_res, tuple):
    valid, msg = (_pv_res + (None,))[:2]  # tolerate 1- or 2-element tuples
else:
    valid, msg = bool(_pv_res), None

if not valid:
    st.error(msg or "Invalid parameters.")
    st.stop()

model = PieceWiseLinearIRM(**params)

# ── Compute current rate and checkpoints ────────────────────────────────────
try:
    current_rate = model.calculate_rate(utilization)
    st.metric("💰 Borrow Rate", f"{current_rate:.4%}")
except Exception as e:
    st.error(f"Could not compute rate: {e}")
    st.stop()

# Key checkpoints for sanity
r_at_0 = model.calculate_rate(0.0)
eps_left = u_opt - 1e-9 if u_opt > 0 else 0.0
eps_right = u_opt + 1e-9 if u_opt < 1.0 else 1.0
r_left = model.calculate_rate(eps_left)
r_right = model.calculate_rate(eps_right)
r_at_1 = model.calculate_rate(1.0)

# ── Plot curve ──────────────────────────────────────────────────────────────
st.subheader("📊 Rate Curve")
U = np.linspace(0.0, 1.0, 501)
R = np.array([model.calculate_rate(u) for u in U])

fig, ax = plt.subplots()
ax.plot(U, R, linewidth=2, label="Piece-wise linear r(u)")
ax.axvline(u_opt, linestyle="--", linewidth=1, label="u_opt (kink)")
ax.axvline(utilization, linestyle=":", linewidth=1, label="current u")
ax.axhline(current_rate, linestyle=":", linewidth=1, label="current r")

if show_construction:
    # left line to kink
    ax.plot([0, u_opt], [r_at_0, r_left], linestyle="--", linewidth=1)
    # right line from kink to 1
    ax.plot([u_opt, 1.0], [r_right, r_at_1], linestyle="--", linewidth=1)

ax.set_xlim(0, 1)
ax.set_ylim(0, float(np.nanmax(R)) * 1.05 if np.isfinite(np.nanmax(R)) else 1.0)
ax.set_xlabel("Utilization (u)")
ax.set_ylabel("Borrow Rate r(u)")
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

# ── Download params (JSON) ──────────────────────────────────────────────────
export = {"type": "piecewise_linear", "params": params}
st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps(export, indent=2),
    file_name="piecewise_linear_irm_params.json",
    mime="application/json",
)

