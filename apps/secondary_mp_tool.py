import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interest_rate_models.secondary_monetary_policy import SecondaryMonetaryPolicy

st.set_page_config(layout="centered", page_title="Secondary Monetary Policy")
st.title("📊 Secondary Monetary Policy Simulator")

st.markdown(
    """
### Short description
Secondary Monetary Policy (SMP) maps utilization and an external reference rate into protocol borrow rate.

### Formula intuition
The model blends:
- a **low-utilization floor effect** (`alpha`),
- a **high-utilization steepness effect** (`beta`),
- and an additive **shift**.

### Parameter intuition
- `u_opt`: target utilization pivot.
- `alpha`: how low rates can stay when utilization is below target.
- `beta`: how fast rates increase above target.
- `shift`: global upward offset.
- `external_rate`: external anchor level.

Reference: [Curve docs — Secondary MP](https://docs.curve.finance/developer/lending/contracts/secondary-mp)
"""
)

st.sidebar.header("Model Parameters")
u_opt = st.sidebar.slider("Target utilization u_opt (%)", 10.0, 99.0, 85.0, 0.5) / 100.0
alpha = st.sidebar.slider("Alpha (low-utilization floor ratio)", 0.01, 0.99, 0.35, 0.01)
beta = st.sidebar.slider("Beta (high-utilization aggressiveness)", 1.01, 99.0, 3.0, 0.1)
shift = st.sidebar.slider("Shift (percentage points)", 0.0, 20.0, 0.0, 0.1) / 100.0
external_rate = st.sidebar.slider("External market rate (%)", 0.0, 100.0, 12.0, 0.1) / 100.0

utilization = st.slider("Current utilization (%)", 1.0, 100.0, 50.0, 0.5) / 100.0

if utilization < 0.02 or utilization > 0.98:
    st.warning("Utilization near 0%/100% can produce sharp moves in output rate.")
if beta > 20:
    st.warning("Very high beta selected; curve can become extremely steep near high utilization.")

# validator preview
denominator = (beta - 1) * u_opt - (1 - u_opt) * (1 - alpha)
if abs(denominator) < 1e-6:
    st.error("Invalid parameter combination: denominator near zero. Adjust alpha/beta/u_opt.")
    st.stop()

try:
    model = SecondaryMonetaryPolicy(
        u_opt=u_opt,
        alpha=alpha,
        beta=beta,
        shift=shift,
        external_rate=external_rate,
    )
    current_rate = model.calculate_rate(utilization, rate=None)
except Exception as e:
    st.error(f"Could not compute rate: {e}")
    st.stop()

st.metric(label="Borrow rate (APY %)", value=f"{current_rate * 100:.2f}%")

st.subheader("Interest-rate curve")
util_vals = np.linspace(0.01, 1.00, 240)
rate_vals = []
error_count = 0
for u in util_vals:
    try:
        rate_vals.append(model.calculate_rate(float(u), rate=None))
    except Exception:
        rate_vals.append(np.nan)
        error_count += 1

if error_count:
    st.warning(f"{error_count} grid points could not be evaluated; check parameter stability.")

fig, ax = plt.subplots()
ax.plot(util_vals * 100, np.array(rate_vals) * 100, label="Secondary MP curve", color="black", linewidth=2)
ax.axvline(utilization * 100, color="red", linestyle="--", label="Current utilization")
ax.axhline(current_rate * 100, color="gray", linestyle=":", label="Current rate")
ax.set_xlabel("Utilization (%)")
ax.set_ylabel("Borrow rate (APY %)")
ax.grid(True, linestyle=':', color='gray')
ax.legend()
st.pyplot(fig, clear_figure=True)

st.download_button(
    "⬇️ Download params JSON",
    data=json.dumps(
        {
            "type": "secondary_monetary_policy",
            "params": {
                "u_opt": u_opt,
                "alpha": alpha,
                "beta": beta,
                "shift": shift,
                "external_rate": external_rate,
            },
        },
        indent=2,
    ),
    file_name="secondary_mp_params.json",
    mime="application/json",
)
