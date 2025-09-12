# smp_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interest_rate_models.secondary_monetary_policy import SecondaryMonetaryPolicy

# App configuration
st.set_page_config(layout="centered", page_title="Secondary Monetary Policy")

# Title
st.title("📈 Secondary Monetary Policy Simulator")

# Introduction
st.markdown(
    """
    The **Secondary Monetary Policy (SMP)** is a utilization-based interest rate model used in Curve's Lending AMM architecture. 
    It is designed to respond **smoothly** to supply and demand imbalances, ensuring capital-efficient rates in both low and high utilization regimes.
    The SMP determines the interest rate `r(u)` as a nonlinear function of utilization `u`, blending an **external market rate** with protocol-specific incentives to ensure competitiveness and stability.

    ℹ️ Read the [official Curve SMP documentation](https://docs.curve.fi/lending/contracts/secondary-mp/) for in-depth details.
    """
)

# Sidebar — Parameter input
st.sidebar.header("Model Parameters")

u_opt = st.sidebar.slider("Target Utilization (u_opt)", 0.1, 0.99, 0.85,
                          help="The ideal utilization level (e.g., 85%). Below this, supply is in excess; above it, borrowing demand dominates.")
alpha = st.sidebar.slider("Alpha (Minimum ratio)", 0.01, 0.99, 0.35,
                          help="Defines the minimum share of the external rate to be passed to borrowers when utilization is low.")
beta = st.sidebar.slider("Beta (Aggressiveness)", 1.01, 99.0, 3.0,
                         help="Controls how aggressively the rate increases as utilization exceeds u_opt. Higher beta → steeper curve.")
shift = st.sidebar.slider("Rate Shift", 0.0, 0.1, 0.00,
                          help="Adds a constant to the output rate. Used to adjust for protocol-level base rate incentives.")
external_rate = st.sidebar.slider(
    "External Market Rate",
    min_value=0.00,
    max_value=0.50,
    value=0.12,
    step=0.001,
    format="%.3f",
    help="Benchmark external interest rate (e.g., CeFi yield or base AMM return)."
)

# Main control — utilization
utilization = st.slider("📌 Current Market Utilization", 0.01, 1.00, 0.5,
                        help="Set the current market utilization to calculate the SMP rate.")

# Model initialization
try:
    model = SecondaryMonetaryPolicy(
        u_opt=u_opt,
        alpha=alpha,
        beta=beta,
        shift=shift,
        external_rate=external_rate
    )
    current_rate = model.calculate_rate(utilization, rate=None)
    st.metric(label="💰 Calculated Borrow Rate", value=f"{current_rate:.4%}")
except Exception as e:
    st.error(f"Could not compute rate: {e}")
    st.stop()

# Rate curve visualization
st.subheader("📊 Interest Rate Curve")

util_vals = np.linspace(0.01, 1.00, 100)
rate_vals = []
for u in util_vals:
    try:
        rate = model.calculate_rate(u, rate=None)
        rate_vals.append(rate)
    except:
        rate_vals.append(np.nan)

fig, ax = plt.subplots()
ax.plot(util_vals, rate_vals, label="SMP Rate Curve", color="black", linewidth=2)
ax.axvline(utilization, color='red', linestyle='--', label="Current Utilization")
ax.axhline(current_rate, color='gray', linestyle=':', label="Current Rate")
ax.set_xlabel("Utilization")
ax.set_ylabel("Borrow Rate")
ax.grid(True, linestyle=':', color='gray')
ax.legend()
st.pyplot(fig)

# Further explanation section
st.markdown(
    """
    ---
    ### 📘 Model Intuition

    - At **low utilization**, the rate remains low to encourage borrowing and avoid unnecessary capital costs.
    - Around **target utilization (`u_opt`)**, the curve gently transitions to higher rates to balance demand and supply.
    - At **high utilization**, the rate increases sharply to discourage further borrowing and attract more liquidity.

    **Why use an external rate?**
    > *The external market rate anchors the SMP to competitive conditions, making sure it doesn’t drift too far from opportunity cost.*

    **Shift parameter?**
    > *It allows the protocol to introduce an incentive buffer or adjust for minimum baseline revenue requirements.*

    ---
    """
)
