import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
ONE = 10**18
MAX_RATE_RAW = 43_959_106_799  # per-second, 1e18-scaled cap


@dataclass
class PolicyInputs:
    rate0_apy: float
    sigma: float
    target_fraction: float
    debt_fraction: float
    price: float
    price_peg: float = 1.0
    extra_const_apy: float = 0.0

    @property
    def rate0_per_sec(self) -> float:
        return (1.0 + self.rate0_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0

    @property
    def extra_per_sec(self) -> float:
        return (1.0 + self.extra_const_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0


def annualize(rate_per_sec: float) -> float:
    return (1.0 + rate_per_sec) ** SECONDS_PER_YEAR - 1.0


def compute_power(price_peg: float, price: float, debt_fraction: float, sigma: float, target_fraction: float) -> float:
    return ((price_peg - price) * ONE) / sigma - (debt_fraction * ONE) / target_fraction


def per_second_rate(P: PolicyInputs, price: float, debt_fraction: float):
    power = compute_power(P.price_peg, price, debt_fraction, P.sigma, P.target_fraction)
    core = P.rate0_per_sec * math.exp(power / ONE)
    final_per_sec = core + P.extra_per_sec
    capped_raw = min(final_per_sec * ONE, MAX_RATE_RAW)
    return capped_raw / ONE, power


def apy_at_point(P: PolicyInputs, price: float, debt_fraction: float) -> float:
    ps, _ = per_second_rate(P, price, debt_fraction)
    return annualize(ps)


@st.cache_data(show_spinner=False)
def apy_grid_cached(
    rate0_apy: float,
    sigma: float,
    target_fraction: float,
    extra_const_apy: float,
    price_peg: float,
    pmin: float,
    pmax: float,
    price_points: int,
    df_points: int,
):
    P = PolicyInputs(
        rate0_apy=rate0_apy,
        sigma=sigma,
        target_fraction=target_fraction,
        debt_fraction=0.0,
        price=1.0,
        price_peg=price_peg,
        extra_const_apy=extra_const_apy,
    )
    prices = np.linspace(pmin, pmax, price_points)
    dfs = np.linspace(0.0, 1.0, df_points)
    PX, DF = np.meshgrid(prices, dfs)
    Z = np.zeros_like(PX)
    for i in range(DF.shape[0]):
        for j in range(PX.shape[1]):
            Z[i, j] = apy_at_point(P, float(PX[i, j]), float(DF[i, j]))
    return prices, dfs, Z


st.set_page_config(page_title="crvUSD Monetary Policy", layout="wide")
st.title("🏦 crvUSD Monetary Policy Simulator")

st.markdown(
    """
A concise simulator for Curve's crvUSD monetary-policy core.

**Core formula (docs):**
- `rate = rate0 * exp(power) + extra_const`
- `power` increases when price is below peg and decreases when debt fraction is above target.
- Final per-second rate is capped by contract max rate.

This version intentionally **excludes debt-ceiling utilization multiplier** (per latest policy direction requested).

Reference: [Curve docs — Monetary Policy Overview](https://docs.curve.finance/developer/crvusd/monetary-policy/overview)
"""
)

st.sidebar.header("Inputs")
mode = st.sidebar.radio("Resolution", ["Fast", "Detailed"], horizontal=True)
if mode == "Fast":
    price_points, df_points = 101, 81
else:
    price_points, df_points = 181, 121

_default_rate0_raw = 3_488_077_118
_default_rate0_ps = _default_rate0_raw / ONE
_default_rate0_apy = (1.0 + _default_rate0_ps) ** SECONDS_PER_YEAR - 1.0

rate0_apy = st.sidebar.slider("Baseline rate0 (APY %)", 0.0, 300.0, float(_default_rate0_apy * 100), 0.1) / 100.0
log_sigma_ui = st.sidebar.slider("Price sensitivity log10(σ)", 14.0, 18.0, float(np.log10(7e15)), 0.01)
sigma = float(np.clip(10 ** log_sigma_ui, 1e14, 1e18))
target_fraction = st.sidebar.slider("Target debt fraction (%)", 1.0, 100.0, 20.0, 1.0) / 100.0
debt_fraction = st.sidebar.slider("Current debt fraction (%)", 0.0, 100.0, 5.0, 0.5) / 100.0
price = st.sidebar.slider("crvUSD price", 0.90, 1.10, 1.00, 0.0005, format="%.4f")
price_peg = st.sidebar.number_input("Peg price", 0.90, 1.10, 1.00, 0.0001, format="%.4f")
extra_const_apy = st.sidebar.slider("extra_const (APY %)", 0.0, 300.0, 0.0, 0.1) / 100.0
log_y = st.sidebar.checkbox("Log-scale Y in slices", value=False)

if sigma <= 1.01e14 or sigma >= 9.9e17:
    st.warning("σ is at an extreme. Response can be unusually flat or steep.")
if debt_fraction < 0.01 or debt_fraction > 0.99:
    st.warning("Debt fraction near 0%/100% can produce extreme policy outputs.")

P = PolicyInputs(
    rate0_apy=rate0_apy,
    sigma=sigma,
    target_fraction=target_fraction,
    debt_fraction=debt_fraction,
    price=price,
    price_peg=price_peg,
    extra_const_apy=extra_const_apy,
)

ps, power = per_second_rate(P, P.price, P.debt_fraction)
annual_rate = annualize(ps)
cap_active = (ps * ONE) >= MAX_RATE_RAW - 1

c1, c2, c3 = st.columns(3)
c1.metric("Debt fraction", f"{P.debt_fraction:.2%}")
c2.metric("Borrow rate (APY %)", f"{annual_rate * 100:.2f}%")
c3.metric("Power (core index)", f"{power / ONE:.4f}")
if cap_active:
    st.warning("Rate cap is active at current point.")

st.markdown("---")
st.subheader("APY surface and slices")

prices, dfs, Z = apy_grid_cached(
    P.rate0_apy,
    P.sigma,
    P.target_fraction,
    P.extra_const_apy,
    P.price_peg,
    0.95,
    1.05,
    price_points,
    df_points,
)

PX, DF = np.meshgrid(prices, dfs)
fig = go.Figure()
fig.add_surface(
    x=PX,
    y=DF,
    z=Z,
    colorscale="Viridis",
    colorbar=dict(title="APY"),
    hovertemplate="price=%{x:.4f}<br>DebtFraction=%{y:.2%}<br>APY=%{z:.2%}<extra></extra>",
)
fig.add_scatter3d(x=[P.price], y=[P.debt_fraction], z=[annual_rate], mode="markers", marker=dict(size=6, color="red"), name="current")
fig.update_layout(
    scene=dict(xaxis_title="price", yaxis_title="DebtFraction", zaxis_title="APY"),
    margin=dict(l=0, r=0, t=10, b=10),
    height=620,
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

row1, row2 = st.columns(2)
with row1:
    xs = np.linspace(0.95, 1.05, 300)
    ys = [apy_at_point(P, float(x), P.debt_fraction) for x in xs]
    fig1 = go.Figure()
    fig1.add_scatter(x=xs, y=ys, mode="lines", name="APY vs price")
    fig1.add_vline(x=P.price, line_dash="dot")
    fig1.add_hline(y=P.rate0_apy, line_dash="dot")
    fig1.update_layout(height=320, xaxis_title="price", yaxis_title="APY")
    if log_y:
        fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

with row2:
    xs = np.linspace(0.0, 1.0, 300)
    ys = [apy_at_point(P, P.price, float(x)) for x in xs]
    fig2 = go.Figure()
    fig2.add_scatter(x=xs, y=ys, mode="lines", name="APY vs debt fraction")
    fig2.add_vline(x=P.debt_fraction, line_dash="dot")
    fig2.add_vline(x=P.target_fraction, line_dash="dot")
    fig2.update_layout(height=320, xaxis_title="DebtFraction", yaxis_title="APY")
    if log_y:
        fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
