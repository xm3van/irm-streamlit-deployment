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


def power_terms(price_peg: float, price: float, debt_fraction: float, sigma: float, target_fraction: float):
    price_term = ((price_peg - price) * ONE) / sigma
    debt_term = debt_fraction / target_fraction
    power = price_term - debt_term
    return price_term, debt_term, power


def per_second_rate(P: PolicyInputs, price: float, debt_fraction: float):
    price_term, debt_term, power = power_terms(P.price_peg, price, debt_fraction, P.sigma, P.target_fraction)
    core = P.rate0_per_sec * math.exp(power)
    final_per_sec = core + P.extra_per_sec
    capped_raw = min(final_per_sec * ONE, MAX_RATE_RAW)
    return capped_raw / ONE, price_term, debt_term, power


def apy_at_point(P: PolicyInputs, price: float, debt_fraction: float) -> float:
    ps, _, _, _ = per_second_rate(P, price, debt_fraction)
    return annualize(ps)


def power_at_point(P: PolicyInputs, price: float, debt_fraction: float) -> float:
    _, _, _, power = per_second_rate(P, price, debt_fraction)
    return power


@st.cache_data(show_spinner=False)
def grids_cached(
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
    Z_apy = np.zeros_like(PX)
    Z_power = np.zeros_like(PX)

    for i in range(DF.shape[0]):
        for j in range(PX.shape[1]):
            px = float(PX[i, j])
            df = float(DF[i, j])
            Z_apy[i, j] = apy_at_point(P, px, df)
            Z_power[i, j] = power_at_point(P, px, df)

    return prices, dfs, PX, DF, Z_apy, Z_power


st.set_page_config(page_title="crvUSD Monetary Policy", layout="wide")
st.title("🏦 crvUSD Monetary Policy Simulator")

st.markdown(
    r"""
### Short description
This app follows the **crvUSD monetary policy overview** and shows how rate reacts to price depeg and debt ratio.

### Formula (docs-aligned core)
\[
\text{rate}_{sec}=\text{rate0}\cdot e^{\text{power}} + \text{extra\_const}
\]
\[
\text{power} = \underbrace{\frac{\text{peg} - \text{price}}{\sigma}}_{\text{price term}} - \underbrace{\frac{\text{DebtFraction}}{\text{TargetFraction}}}_{\text{debt term}}
\]
Final per-second rate is capped by contract max rate and then annualized for APY display.

### Intuition
- **Price term**: below-peg price increases power (higher rates).
- **Debt term**: higher debt ratio vs target decreases power (lower rates).
- **Sigma** controls price sensitivity: smaller sigma = sharper response.

> Requested behavior kept: debt-ceiling utilization multiplier is excluded.

Reference: [Curve docs — Monetary Policy Overview](https://docs.curve.finance/developer/crvusd/monetary-policy/overview)
"""
)

st.sidebar.header("Inputs")
quality = st.sidebar.radio("Grid quality", ["Fast", "Detailed"], horizontal=True)
if quality == "Fast":
    price_points, df_points = 101, 81
else:
    price_points, df_points = 181, 121

_default_rate0_raw = 3_488_077_118
_default_rate0_ps = _default_rate0_raw / ONE
_default_rate0_apy = (1.0 + _default_rate0_ps) ** SECONDS_PER_YEAR - 1.0

# range audit + calibrated defaults
rate0_apy = st.sidebar.slider("Baseline rate0 (APY %)", 0.0, 100.0, float(_default_rate0_apy * 100), 0.1) / 100.0
log_sigma_ui = st.sidebar.slider("Price sensitivity log10(σ)", 14.0, 18.0, float(np.log10(7e15)), 0.01)
sigma = float(np.clip(10 ** log_sigma_ui, 1e14, 1e18))
target_fraction = st.sidebar.slider("Target debt fraction (%)", 1.0, 80.0, 20.0, 1.0) / 100.0
debt_fraction = st.sidebar.slider("Current debt fraction (%)", 0.0, 100.0, 5.0, 0.5) / 100.0
price = st.sidebar.slider("crvUSD price", 0.90, 1.10, 1.00, 0.0005, format="%.4f")
price_peg = st.sidebar.number_input("Peg price", 0.95, 1.05, 1.00, 0.0001, format="%.4f")
extra_const_apy = st.sidebar.slider("extra_const (APY %)", 0.0, 50.0, 0.0, 0.1) / 100.0
log_y = st.sidebar.checkbox("Log-scale Y in APY line charts", value=False)

# range-audit warnings
if rate0_apy > 0.5:
    st.warning("rate0 > 50% APY is unusual operationally; verify this scenario is intentional.")
if sigma < 3e15 or sigma > 2e16:
    st.warning("σ is outside the typical operational band (~3e15 to ~2e16); sensitivity may be unrealistic.")
if target_fraction < 0.05 or target_fraction > 0.5:
    st.warning("Target fraction outside ~5% to 50% can make debt term dominate or vanish.")
if extra_const_apy > 0.2:
    st.warning("extra_const > 20% APY strongly shifts the full surface upward.")

P = PolicyInputs(
    rate0_apy=rate0_apy,
    sigma=sigma,
    target_fraction=target_fraction,
    debt_fraction=debt_fraction,
    price=price,
    price_peg=price_peg,
    extra_const_apy=extra_const_apy,
)

ps, price_term, debt_term, power = per_second_rate(P, P.price, P.debt_fraction)
annual_rate = annualize(ps)
cap_active = (ps * ONE) >= MAX_RATE_RAW - 1

# metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Borrow rate (APY %)", f"{annual_rate * 100:.2f}%")
m2.metric("Power", f"{power:.4f}")
m3.metric("Price term", f"{price_term:.4f}")
m4.metric("Debt term", f"{debt_term:.4f}")
m5.metric("Debt fraction", f"{P.debt_fraction:.2%}")
if cap_active:
    st.warning("Rate cap is active at the current point.")

# decomposition chart
st.subheader("Power decomposition at current point")
waterfall = go.Figure(
    go.Waterfall(
        name="power",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["+ Price term", "- Debt term", "Power"],
        text=[f"{price_term:.4f}", f"{-debt_term:.4f}", f"{power:.4f}"],
        y=[price_term, -debt_term, power],
    )
)
waterfall.update_layout(height=300, yaxis_title="Contribution to power")
st.plotly_chart(waterfall, use_container_width=True, config={"displaylogo": False})

# rationale for visuals
st.info(
    "**Why these views?**\n"
    "1) Decomposition shows *why* rate moved (price vs debt term).\n"
    "2) Surface/heatmap shows nonlinear interaction across state space.\n"
    "3) 2D slices are easiest for policy tuning and sanity checks."
)

prices, dfs, PX, DF, Z_apy, Z_power = grids_cached(
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

tabs = st.tabs(["3D APY surface", "Heatmaps", "Policy slices", "Range audit"])

with tabs[0]:
    fig3d = go.Figure()
    fig3d.add_surface(x=PX, y=DF, z=Z_apy, colorscale="Viridis", colorbar=dict(title="APY"))
    df_line = np.linspace(0, 1, 120)
    z_df_line = [apy_at_point(P, P.price, float(d)) for d in df_line]
    fig3d.add_scatter3d(x=[P.price] * len(df_line), y=df_line, z=z_df_line, mode="lines", name="slice @ current price", line=dict(width=6))
    p_line = np.linspace(0.95, 1.05, 160)
    z_p_line = [apy_at_point(P, float(pp), P.debt_fraction) for pp in p_line]
    fig3d.add_scatter3d(x=p_line, y=[P.debt_fraction] * len(p_line), z=z_p_line, mode="lines", name="slice @ current debt frac", line=dict(width=6))
    fig3d.add_scatter3d(x=[P.price], y=[P.debt_fraction], z=[annual_rate], mode="markers", marker=dict(size=7, color="red"), name="current")
    fig3d.update_layout(scene=dict(xaxis_title="price", yaxis_title="DebtFraction", zaxis_title="APY"), height=620)
    st.plotly_chart(fig3d, use_container_width=True, config={"displaylogo": False})

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        fig_h1 = go.Figure(go.Heatmap(z=Z_apy, x=prices, y=dfs, colorscale="Viridis", colorbar=dict(title="APY")))
        fig_h1.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", marker=dict(size=8, color="red"), name="current")
        fig_h1.update_layout(height=320, title="APY heatmap", xaxis_title="price", yaxis_title="DebtFraction")
        st.plotly_chart(fig_h1, use_container_width=True, config={"displaylogo": False})
    with c2:
        fig_h2 = go.Figure(go.Heatmap(z=Z_power, x=prices, y=dfs, colorscale="RdBu", reversescale=True, colorbar=dict(title="Power")))
        fig_h2.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", marker=dict(size=8, color="black"), name="current")
        fig_h2.update_layout(height=320, title="Power heatmap", xaxis_title="price", yaxis_title="DebtFraction")
        st.plotly_chart(fig_h2, use_container_width=True, config={"displaylogo": False})

with tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        xs = np.linspace(0.95, 1.05, 301)
        ys = [apy_at_point(P, float(x), P.debt_fraction) for x in xs]
        fig1 = go.Figure()
        fig1.add_scatter(x=xs, y=ys, mode="lines", name="APY vs price")
        fig1.add_vline(x=P.price, line_dash="dot")
        fig1.update_layout(height=300, xaxis_title="price", yaxis_title="APY")
        if log_y:
            fig1.update_yaxes(type="log")
        st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})
    with c2:
        xs = np.linspace(0.0, 1.0, 301)
        ys = [apy_at_point(P, P.price, float(x)) for x in xs]
        fig2 = go.Figure()
        fig2.add_scatter(x=xs, y=ys, mode="lines", name="APY vs debt fraction")
        fig2.add_vline(x=P.debt_fraction, line_dash="dot")
        fig2.add_vline(x=P.target_fraction, line_dash="dot", line_color="gray")
        fig2.update_layout(height=300, xaxis_title="DebtFraction", yaxis_title="APY")
        if log_y:
            fig2.update_yaxes(type="log")
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    with c3:
        xs = np.linspace(0.95, 1.05, 301)
        y_price = [((P.price_peg - float(x)) * ONE) / P.sigma for x in xs]
        y_debt = np.full_like(xs, P.debt_fraction / P.target_fraction)
        fig3 = go.Figure()
        fig3.add_scatter(x=xs, y=y_price, mode="lines", name="price term")
        fig3.add_scatter(x=xs, y=y_debt, mode="lines", name="debt term")
        fig3.add_vline(x=P.price, line_dash="dot")
        fig3.update_layout(height=300, xaxis_title="price", yaxis_title="Term value")
        st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})

with tabs[3]:
    st.markdown(
        """
### Parameter range audit (practical guidance)

- **rate0 (APY%)**: slider uses `0..100%`. Typical operation is much lower; use high values only for stress tests.
- **log10(σ)**: slider uses `14..18` to match contract bounds. Typical working zone is around `15.5..16.3`.
- **TargetFraction**: slider uses `1..80%`; practical monitoring is often around `10..40%`.
- **DebtFraction**: `0..100%` is correct by definition.
- **price**: `0.90..1.10` captures relevant peg stress; widen only for extreme scenario testing.
- **extra_const**: slider uses `0..50%`; high values can dominate model dynamics.

If you want, next pass can add a **"Recommended mode"** toggle that hard-clamps to tighter production ranges.
"""
    )

# iso-apy quick guidance
st.markdown("---")
st.subheader("Iso-APY guidance (core approximation)")
xs = np.linspace(0.95, 1.05, 301)
iso_target_apy = float(max(P.rate0_apy, 0.12))
rps = deannualize(iso_target_apy)
power_target = math.log(max(rps, 1e-24) / max(P.rate0_per_sec, 1e-24))
iso_df = P.target_fraction * ((P.price_peg - xs) * (ONE / P.sigma) - power_target)
iso_df = np.clip(iso_df, 0.0, 1.0)
fig_iso = go.Figure()
fig_iso.add_scatter(x=xs, y=iso_df, mode="lines", name=f"Target APY {iso_target_apy:.2%}", line=dict(width=3))
fig_iso.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", marker=dict(size=9, color="red"), name="current")
fig_iso.update_layout(height=320, xaxis_title="price", yaxis_title="Implied debt fraction")
st.plotly_chart(fig_iso, use_container_width=True, config={"displaylogo": False})
