# Semi-Log IRM — Streamlit App
# Save as: semilog_irm_app.py
# Run with:  python -m streamlit run semilog_irm_app.py

import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ── Import SemiLogIRM from your project ─────────────────────────────────────
# Try a package-style import first, then fallback to a local module.
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from interest_rate_models.semilog_irm import SemiLogIRM  # type: ignore
except Exception as e_primary:  # pragma: no cover
    try:
        from interest_rate_models.semilog_irm import SemiLogIRM  # type: ignore
    except Exception as e_fallback:
        st.error(
            "Could not import SemiLogIRM from either\n\n"
            " • interest_rate_models.semilog_irm\n"
            " • semilog_irm (local next to this app)\n\n"
            f"Primary error: {e_primary}\nFallback error: {e_fallback}"
        )
        st.stop()

# ── Streamlit UI ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Semi‑Log IRM", layout="centered")
st.title("🧮 Semi‑Log IRM Simulator")

st.markdown(
    r"""
This app explores a **Semi‑Log** interest‑rate model where the borrow rate grows
*exponentially* with utilization.

**Definition**  
$$
  r(u) = r_{\min} \cdot \Big(\frac{r_{\max}}{r_{\min}}\Big)^{u}
\;=\; \exp\big(\ln r_{\min} + u\,\ln(\tfrac{r_{\max}}{r_{\min}})\big),\quad u\in[0,1].
$$

> **Units**: Rates are **decimals** (e.g., 0.10 = 10%, 10 = 1000%).
    """
)

# Sidebar — Parameters (sliders)
st.sidebar.header("Parameters")

rate_min = st.sidebar.slider(
    "rate_min",
    min_value=1e-12,
    max_value=1.0,
    value=0.0001,
    step=0.00001,
    format="%.4f",
    help="Minimum (floor) borrow rate r_min as a decimal (e.g., 0.0001 = 0.01%).",
)

# Ensure rate_max > rate_min
min_rate_max = max(rate_min + 1e-6, 0.001)
rate_max = st.sidebar.slider(
    "rate_max",
    min_value=float(min_rate_max),
    max_value=10.0,
    value=float(max(1.0, rate_min * 1000)),
    step=0.01,
    format="%.2f",
    help="Maximum (ceiling) borrow rate r_max as a decimal (e.g., 10 = 1000%).",
)

log_y = st.sidebar.checkbox(
    "Log‑scale Y‑axis",
    value=False,
    help="Recommended for exponential curves to see the whole range clearly.",
)

# Main — utilization slider
utilization = st.slider(
    "📌 Current Utilization",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.001,
    help="Set current utilization u in [0, 1] to evaluate r(u).",
)

# ── Validate + instantiate model ────────────────────────────────────────────
params = {"rate_min": rate_min, "rate_max": rate_max}

# Accept validators that return bool OR (bool, message)
try:
    _pv_res = SemiLogIRM.param_validator(params)
except Exception as _pv_exc:  # If validator itself errors, show a helpful message
    st.error(f"Parameter validation raised an exception: {_pv_exc}")
    st.stop()

if isinstance(_pv_res, tuple):
    valid, msg = (_pv_res + (None,))[:2]  # tolerate 1- or 2-element tuples
else:
    valid, msg = bool(_pv_res), None

if not valid:
    st.error(msg or "Invalid parameters: require rate_min>0, rate_max>0, rate_max>rate_min.")
    st.stop()

model = SemiLogIRM(rate_min=rate_min, rate_max=rate_max)

# ── Compute current rate and checkpoints ────────────────────────────────────
try:
    current_rate = model.calculate_rate(utilization)
    st.metric("💰 Borrow Rate", f"{current_rate:.4%}")
except Exception as e:
    st.error(f"Could not compute rate: {e}")
    st.stop()

# ── Plot curve ──────────────────────────────────────────────────────────────
st.subheader("📊 Rate Curve")
U = np.linspace(0.0, 1.0, 501)
R = np.array([model.calculate_rate(u) for u in U])

fig, ax = plt.subplots()
ax.plot(U, R, linewidth=2, label="Semi‑Log r(u)")
ax.axvline(utilization, linestyle=":", linewidth=1, label="current u")
ax.axhline(current_rate, linestyle=":", linewidth=1, label="current r")

# Axes formatting
ax.set_xlim(0, 1)
if np.isfinite(np.nanmin(R)) and np.isfinite(np.nanmax(R)):
    ymin = max(1e-8, float(np.nanmin(R)) * 0.95)
    ymax = float(np.nanmax(R)) * 1.05
    ax.set_ylim(ymin, ymax)
if log_y:
    ax.set_yscale("log")
ax.set_xlabel("Utilization (u)")
ax.set_ylabel("Borrow Rate r(u)")
ax.grid(True, linestyle=":", linewidth=0.5)
ax.legend(loc="upper left")
st.pyplot(fig, clear_figure=True)

# ── Download params (JSON) ──────────────────────────────────────────────────
export = {"type": "semi_log", "params": params}
st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps(export, indent=2),
    file_name="semilog_irm_params.json",
    mime="application/json",
)

