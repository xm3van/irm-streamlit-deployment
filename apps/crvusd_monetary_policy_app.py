import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
ONE = 10**18
MAX_RATE_RAW = 43_959_106_799  # per-second, 1e18-scaled cap (~300% APY equivalent)


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


def deannualize(apy: float) -> float:
    return (1.0 + apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0


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
    return prices, dfs, PX, DF, Z


st.set_page_config(page_title="crvUSD Monetary Policy", layout="wide")
st.title("🏦 crvUSD Monetary Policy Simulator")

st.markdown(
    """
### Short description
This simulator visualizes Curve's crvUSD monetary-policy core using price and debt-fraction state variables.

### Formula
\[
\text{rate}_{sec}=\text{rate0}\cdot e^{\text{power}} + \text{extra\_const}
\]
where
\[
\text{power}=\frac{\text{peg}-\text{price}}{\sigma} - \frac{\text{DebtFraction}}{\text{TargetFraction}}
\]
Then APY is annualized from per-second rate, and capped by contract max rate.

### Explanation
- Price below peg increases `power` → raises rate.
- Higher debt fraction relative to target lowers `power` → lowers rate.
- `extra_const` shifts the full curve upward.

### Parameter intuition
- `rate0`: baseline level when pressure is neutral.
- `sigma`: price sensitivity (smaller = more reactive).
- `target_fraction`: debt-fraction normalization point.
- `extra_const`: additive APY offset.

> Note: Per your request, the debt-ceiling utilization multiplier is removed from this app.

Reference: [Curve docs — Monetary Policy Overview](https://docs.curve.finance/developer/crvusd/monetary-policy/overview)
"""
)

st.sidebar.header("Inputs")
quality = st.sidebar.radio("Grid quality", ["Fast", "Detailed"], horizontal=True)
if quality == "Fast":
    price_points, df_points = 111, 81
else:
    price_points, df_points = 191, 131

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
log_y = st.sidebar.checkbox("Log-scale Y for 2D panels", value=False)

if sigma <= 1.01e14 or sigma >= 9.9e17:
    st.warning("σ is at an extreme. Response can be unusually flat or steep.")
if debt_fraction < 0.01 or debt_fraction > 0.99:
    st.warning("Debt fraction near 0%/100% can produce extreme outputs.")

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

# top metrics and local sensitivities
col1, col2, col3, col4 = st.columns(4)
col1.metric("Borrow rate (APY %)", f"{annual_rate * 100:.2f}%")
col2.metric("Power index", f"{power / ONE:.4f}")
col3.metric("Debt fraction", f"{P.debt_fraction:.2%}")
col4.metric("σ", f"{P.sigma:.2e}")

if cap_active:
    st.warning("Rate cap is active at the current point.")

# quick finite-difference sensitivity chips
ap_now = apy_at_point(P, P.price, P.debt_fraction)
ap_p_up = apy_at_point(P, P.price + 0.001, P.debt_fraction)
ap_p_dn = apy_at_point(P, P.price - 0.001, P.debt_fraction)
ap_df_up = apy_at_point(P, P.price, min(P.debt_fraction + 0.01, 1.0))
ap_df_dn = apy_at_point(P, P.price, max(P.debt_fraction - 0.01, 0.0))

s1, s2, s3, s4 = st.columns(4)
s1.metric("+0.001 price", f"{(ap_p_up - ap_now) * 1e4:+.1f} bps")
s2.metric("-0.001 price", f"{(ap_p_dn - ap_now) * 1e4:+.1f} bps")
s3.metric("+1pp debt frac", f"{(ap_df_up - ap_now) * 1e4:+.1f} bps")
s4.metric("-1pp debt frac", f"{(ap_df_dn - ap_now) * 1e4:+.1f} bps")

st.markdown("---")
prices, dfs, PX, DF, Z = apy_grid_cached(
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

# 3D surface
st.subheader("3D Surface: APY(price, debt fraction)")
fig3d = go.Figure()
fig3d.add_surface(
    x=PX,
    y=DF,
    z=Z,
    colorscale="Viridis",
    colorbar=dict(title="APY"),
    hovertemplate="price=%{x:.4f}<br>DebtFraction=%{y:.2%}<br>APY=%{z:.2%}<extra></extra>",
)
# crosshair slices
df_line = np.linspace(0, 1, 140)
z_df_line = [apy_at_point(P, P.price, float(d)) for d in df_line]
fig3d.add_scatter3d(x=[P.price] * len(df_line), y=df_line, z=z_df_line, mode="lines", name="slice @ current price", line=dict(width=6))

p_line = np.linspace(0.95, 1.05, 200)
z_p_line = [apy_at_point(P, float(pp), P.debt_fraction) for pp in p_line]
fig3d.add_scatter3d(x=p_line, y=[P.debt_fraction] * len(p_line), z=z_p_line, mode="lines", name="slice @ current debt frac", line=dict(width=6))

fig3d.add_scatter3d(x=[P.price], y=[P.debt_fraction], z=[annual_rate], mode="markers", marker=dict(size=7, color="red"), name="current")
fig3d.update_layout(
    scene=dict(xaxis_title="price", yaxis_title="DebtFraction", zaxis_title="APY"),
    height=630,
    margin=dict(l=0, r=0, t=10, b=10),
)
st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})

st.markdown("---")
# supporting 2D charts
r1, r2, r3 = st.columns(3)

with r1:
    st.markdown("#### APY vs Price")
    xs = np.linspace(0.95, 1.05, 301)
    fig1 = go.Figure()
    for df in (0.0, 0.05, 0.10, 0.20, 0.40, 0.60):
        ys = [apy_at_point(P, float(x), df) for x in xs]
        fig1.add_scatter(x=xs, y=ys, mode="lines", name=f"DF {df:.0%}")
    fig1.add_vline(x=P.price, line_dash="dot")
    fig1.add_hline(y=P.rate0_apy, line_dash="dot", line_color="gray")
    fig1.add_scatter(x=[P.price], y=[annual_rate], mode="markers", marker=dict(size=8), name="current")
    fig1.update_layout(height=320, xaxis_title="price", yaxis_title="APY", legend=dict(orientation="h", y=1.02))
    if log_y:
        fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

with r2:
    st.markdown("#### APY vs Debt Fraction")
    xs = np.linspace(0.0, 1.0, 301)
    fig2 = go.Figure()
    for tgt, label in [
        (max(P.target_fraction * 0.5, 0.01), "Target ×0.5"),
        (P.target_fraction, "Target current"),
        (min(P.target_fraction * 1.5, 1.0), "Target ×1.5"),
    ]:
        P_tmp = PolicyInputs(P.rate0_apy, P.sigma, float(tgt), P.debt_fraction, P.price, P.price_peg, P.extra_const_apy)
        ys = [apy_at_point(P_tmp, P.price, float(df)) for df in xs]
        fig2.add_scatter(x=xs, y=ys, mode="lines", name=label)
    fig2.add_vline(x=P.debt_fraction, line_dash="dot")
    fig2.add_vline(x=P.target_fraction, line_dash="dot", line_color="gray")
    fig2.add_scatter(x=[P.debt_fraction], y=[annual_rate], mode="markers", marker=dict(size=8), name="current")
    fig2.update_layout(height=320, xaxis_title="DebtFraction", yaxis_title="APY", legend=dict(orientation="h", y=1.02))
    if log_y:
        fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

with r3:
    st.markdown("#### Heatmap + baseline frontier")
    df_curve = P.target_fraction * np.maximum(0.0, (P.price_peg - prices)) * (ONE / P.sigma)
    df_curve = np.clip(df_curve, 0.0, 1.0)
    figh = go.Figure()
    figh.add_trace(go.Heatmap(z=Z, x=prices, y=dfs, colorscale="Viridis", colorbar=dict(title="APY")))
    figh.add_scatter(x=prices, y=df_curve, mode="lines", name="core baseline frontier", line=dict(color="white", dash="dash", width=3))
    figh.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", marker=dict(size=9, color="red"), name="current")
    figh.update_layout(height=320, xaxis_title="price", yaxis_title="DebtFraction", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(figh, use_container_width=True, config={"displaylogo": False})

st.markdown("---")
st.subheader("Iso-APY guidance (core approximation)")
xs = np.linspace(0.95, 1.05, 301)
iso_target_apy = float(max(P.rate0_apy, 0.12))
rps = deannualize(iso_target_apy)
power_target = ONE * math.log(max(rps, 1e-24) / max(P.rate0_per_sec, 1e-24))
iso_df = P.target_fraction * ((P.price_peg - xs) * (ONE / P.sigma) - (power_target / ONE))
iso_df = np.clip(iso_df, 0.0, 1.0)

fig_iso = go.Figure()
fig_iso.add_scatter(x=xs, y=iso_df, mode="lines", name=f"Target APY {iso_target_apy:.2%}", line=dict(width=3))
fig_iso.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", marker=dict(size=9, color="red"), name="current")
fig_iso.update_layout(height=320, xaxis_title="price", yaxis_title="Implied debt fraction")
st.plotly_chart(fig_iso, use_container_width=True, config={"displaylogo": False})
