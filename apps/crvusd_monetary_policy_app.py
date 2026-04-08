import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import streamlit as st

SECONDS_PER_YEAR = 365 * 24 * 60 * 60
ONE = 10**18
MAX_RATE_RAW = 43_959_106_799  # per-second, 1e18-scaled cap (~300% APY equivalent)
PEG_PRICE = 1.0


@dataclass
class PolicyInputs:
    rate0_apy: float
    sigma: float
    target_fraction: float
    debt_fraction: float
    price: float
    price_peg: float = PEG_PRICE
    extra_const_apy: float = 0.0

    @property
    def rate0_per_sec(self) -> float:
        return (1.0 + self.rate0_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0

    @property
    def extra_per_sec(self) -> float:
        return (1.0 + self.extra_const_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0


def annualize(rate_per_sec: float) -> float:
    return (1.0 + rate_per_sec) ** SECONDS_PER_YEAR - 1.0


def power_terms(price_peg: float, price: float, debt_fraction: float, sigma: float, target_fraction: float):
    price_term = ((price_peg - price) * ONE) / sigma
    debt_term = debt_fraction / target_fraction
    power = price_term - debt_term
    return price_term, debt_term, power


def per_second_rate(P: PolicyInputs, price: float, debt_fraction: float):
    price_term, debt_term, power = power_terms(
        P.price_peg, price, debt_fraction, P.sigma, P.target_fraction
    )

    max_per_sec = MAX_RATE_RAW / ONE
    base = P.rate0_per_sec
    extra = P.extra_per_sec

    # If extra already reaches or exceeds the cap, we are capped regardless of power.
    if extra >= max_per_sec:
        return max_per_sec, price_term, debt_term, power

    # Remaining room before hitting the cap.
    remaining = max_per_sec - extra

    # If base is zero or negative, the exponential core contributes nothing meaningful.
    if base <= 0:
        final_per_sec = extra
        return min(final_per_sec, max_per_sec), price_term, debt_term, power

    # Solve for the power level at which:
    # base * exp(power) + extra = max_per_sec
    # => power = log((max_per_sec - extra) / base)
    power_cap_threshold = math.log(remaining / base)

    # If power is beyond that threshold, the result is capped anyway.
    if power >= power_cap_threshold:
        return max_per_sec, price_term, debt_term, power

    # Optional lower clamp for numerical stability.
    # exp(-745) is already effectively zero in double precision.
    safe_power = max(power, -745.0)

    core = base * math.exp(safe_power)
    final_per_sec = core + extra

    return min(final_per_sec, max_per_sec), price_term, debt_term, power

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
        price=PEG_PRICE,
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

    # Numerical local sensitivities
    d_apy_d_price = np.gradient(Z_apy, prices, axis=1)
    d_apy_d_df = np.gradient(Z_apy, dfs, axis=0)

    # Convert from decimal-APY derivatives into percentage-point interpretations:
    # price axis: 0.01 = 1 cent
    # debt fraction axis: 0.01 = 1 percentage point
    sensitivity_price_pp_per_1c = d_apy_d_price * 100 * 0.01
    sensitivity_df_pp_per_1pp = d_apy_d_df * 100 * 0.01
    return (
        prices,
        dfs,
        PX,
        DF,
        Z_apy,
        Z_power,
        sensitivity_price_pp_per_1c,
        sensitivity_df_pp_per_1pp,
    )


st.set_page_config(page_title="crvUSD Monetary Policy", layout="wide")
st.title("🏦 crvUSD Monetary Policy Simulator")

st.markdown(
    r"""
This app shows the **crvUSD monetary policy** and shows how the borrow rate reacts to price depeg and debt ratio.

Formula:
$$
\text{rate}_{sec}=\text{rate0}\cdot e^{\text{power}} + \text{extra\_const}
$$
$$
\text{power} = \underbrace{\frac{\text{peg} - \text{price}}{\sigma}}_{\text{price term}} - \underbrace{\frac{\text{DebtFraction}}{\text{TargetFraction}}}_{\text{debt term}}
$$
Final per-second rate is capped by the contract max rate and then annualized for APY display.

Intuition:
- **Price term**: below-peg price increases power and pushes rates higher.
- **Debt term**: higher debt ratio relative to target lowers power and pushes rates lower.
- **Sigma** controls price sensitivity: smaller sigma means a sharper response.

Reference: [Curve docs — Monetary Policy Overview](https://docs.curve.finance/developer/crvusd/monetary-policy/overview)
"""
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Inputs")

price_points, df_points = 141, 101

_default_rate0_raw = 3_488_077_118
_default_rate0_ps = _default_rate0_raw / ONE
_default_rate0_apy = (1.0 + _default_rate0_ps) ** SECONDS_PER_YEAR - 1.0

rate0_apy = st.sidebar.slider(
    "Baseline rate0 (APY %)",
    0.0,
    25.0,
    float(_default_rate0_apy * 100),
    0.1,
    help=(
        "Definition: the baseline borrow rate before the policy reacts to market conditions.\n\n"
        "Interpretation: raising rate0 shifts the whole policy upward. Even if price and debt fraction stay unchanged, "
        "the system starts from a higher base rate.\n\n"
        "Practical range: low single digits are typical for normal operation; high values are mainly useful for stress testing."
    ),
) / 100.0

sigma_ui = st.sidebar.slider(
    "Sigma",
    0.0001,
    1.0000,
    0.0080,
    0.0001,
    format="%.4f",
)
sigma = sigma_ui * ONE

target_fraction = st.sidebar.slider(
    "Target debt fraction (%)",
    1.0,
    100.0,
    20.0,
    1.0,
    help=(
        "Definition: the reference debt share used in the policy denominator for the debt term.\n\n"
        "Interpretation: a larger target fraction makes a given debt fraction look less constraining; a smaller target fraction makes the debt term bite sooner.\n\n"
        "Practical range: roughly 10% to 40% is a sensible region to inspect for many scenarios."
    ),
) / 100.0

debt_fraction = st.sidebar.slider(
    "Current debt fraction (%)",
    0.0,
    100.0,
    5.0,
    0.5,
    help=(
        "Definition: the current share of system debt attributed to the market under evaluation.\n\n"
        "Interpretation: higher debt fraction increases the negative debt term and therefore lowers policy power.\n\n"
        "Practical note: this is a state variable, not a tuning parameter. It tells you where the market currently sits in policy space."
    ),
) / 100.0

price = st.sidebar.slider(
    "crvUSD Oracle price",
    0.00,
    2.00,
    1.00,
    0.0005,
    format="%.4f",
    help=(
        "Definition: the observed market price of crvUSD.\n\n"
        "Interpretation: a price below 1.00 raises the price term and pushes rates upward; a price above 1.00 does the opposite.\n\n"
        "Practical range: 0.95 to 1.05 is the most policy-relevant window; the wider range is useful for stress exploration."
    ),
)

extra_const_apy = st.sidebar.slider(
    "extra_const (APY %)",
    0.0,
    50.0,
    0.0,
    0.1,
    help=(
        "Definition: a constant APY add-on applied after the exponential policy component.\n\n"
        "Interpretation: it lifts rates mechanically and can dominate the policy if set too high.\n\n"
        "Practical range: usually small or zero unless you are explicitly testing overlays or buffers."
    ),
) / 100.0

P = PolicyInputs(
    rate0_apy=rate0_apy,
    sigma=sigma,
    target_fraction=target_fraction,
    debt_fraction=debt_fraction,
    price=price,
    price_peg=PEG_PRICE,
    extra_const_apy=extra_const_apy,
)

ps, price_term, debt_term, power = per_second_rate(P, P.price, P.debt_fraction)
annual_rate = annualize(ps)
cap_active = (ps * ONE) >= MAX_RATE_RAW - 1

# =========================
# Top summary metrics
# =========================
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Borrow rate (APY %)", f"{annual_rate * 100:.2f}%")
m2.metric("Power", f"{power:.4f}")
m3.metric("Price term", f"{price_term:.4f}")
m4.metric("Debt term", f"{debt_term:.4f}")
m5.metric("Debt fraction", f"{P.debt_fraction:.2%}")
if cap_active:
    st.warning("Rate cap is active at the current point.")

(
    prices,
    dfs,
    PX,
    DF,
    Z_apy,
    Z_power,
    sensitivity_price_pp_per_1c,
    sensitivity_df_pp_per_1pp,
) = grids_cached(
    P.rate0_apy,
    P.sigma,
    P.target_fraction,
    P.extra_const_apy,
    P.price_peg,
    0.80,
    1.20,
    price_points,
    df_points,
)

# =========================
# Main figure: 3D surface only
# =========================
st.subheader("Central policy view")
st.markdown(
    "**What this shows:** the full APY surface implied by the policy across price and debt fraction states.  \n"
    "**Interpretation:** the height gives the borrow rate. Steeper areas mean the policy reacts more sharply to changes in state. "
    "The red point marks the current state."
)

fig_main = go.Figure()
fig_main.add_surface(
    x=PX,
    y=DF,
    z=Z_apy * 100,
    colorscale="Viridis",
    colorbar=dict(title="APY %"),
    name="APY surface",
)
fig_main.add_scatter3d(
    x=[P.price],
    y=[P.debt_fraction],
    z=[annual_rate * 100],
    mode="markers",
    marker=dict(size=7, color="red"),
    name="current state",
)
fig_main.update_layout(
    height=680,
    scene=dict(
        xaxis_title="price",
        yaxis_title="Debt fraction",
        zaxis_title="APY %",
    ),
)
st.plotly_chart(fig_main, use_container_width=True, config={"displaylogo": False})

# =========================
# Detail tabs
# =========================
tabs = st.tabs([
    "Heatmaps",
    "Contour map / iso-rate lines",
    "Local sensitivity / elasticity",
    "Policy slices",
    "Power decomposition",
])

with tabs[0]:
    st.markdown(
        "**What this shows:** two top-down maps of the policy state space. The left panel shows APY levels and the right panel shows policy power.  \n"
        "**Interpretation:** use the APY heatmap for level intuition and the power heatmap for structural intuition. "
        "The red marker shows the current state."
    )
    c1, c2 = st.columns(2)

    with c1:
        fig_h1 = go.Figure(
            go.Heatmap(
                z=Z_apy * 100,
                x=prices,
                y=dfs,
                colorscale="Viridis",
                colorbar=dict(title="APY %"),
            )
        )
        fig_h1.add_scatter(
            x=[P.price],
            y=[P.debt_fraction],
            mode="markers",
            marker=dict(size=8, color="red"),
            name="current",
        )
        fig_h1.update_layout(
            height=360,
            title="APY heatmap",
            xaxis_title="price",
            yaxis_title="Debt fraction",
        )
        st.plotly_chart(fig_h1, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_h2 = go.Figure(
            go.Heatmap(
                z=Z_power,
                x=prices,
                y=dfs,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Power"),
            )
        )
        fig_h2.add_scatter(
            x=[P.price],
            y=[P.debt_fraction],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="current",
        )
        fig_h2.update_layout(
            height=360,
            title="Power heatmap",
            xaxis_title="price",
            yaxis_title="Debt fraction",
        )
        st.plotly_chart(fig_h2, use_container_width=True, config={"displaylogo": False})

with tabs[1]:
    st.markdown(
        "**What this shows:** contour lines connecting all combinations of price and debt fraction that imply the same APY.  \n"
        "**Interpretation:** if two points lie on the same contour, the policy assigns them the same rate. "
        "Tightly packed contours indicate regions where the rate changes quickly. Their tilt shows the trade-off between price stress and debt stress."
    )

    fig_contour = go.Figure(
        go.Contour(
            z=Z_apy * 100,
            x=prices,
            y=dfs,
            colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=10, color="white")),
            colorbar=dict(title="APY %"),
        )
    )
    fig_contour.add_scatter(
        x=[P.price],
        y=[P.debt_fraction],
        mode="markers",
        marker=dict(size=9, color="red"),
        name="current",
    )
    fig_contour.update_layout(
        height=520,
        xaxis_title="price",
        yaxis_title="Debt fraction",
    )
    st.plotly_chart(fig_contour, use_container_width=True, config={"displaylogo": False})

with tabs[2]:
    st.markdown(
        "**What this shows:** numerical local sensitivity of APY to small movements in price and debt fraction.  \n"
        "**Interpretation:** the left panel is approximately the APY percentage-point change from a **1 cent** move in price; "
        "the right panel is approximately the APY percentage-point change from a **1 percentage-point** move in debt fraction. "
        "Large absolute values mean the policy is locally aggressive."
    )
    c1, c2 = st.columns(2)

    with c1:
        fig_s1 = go.Figure(
            go.Heatmap(
                z=sensitivity_price_pp_per_1c,
                x=prices,
                y=dfs,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Δ APY pp per 1c"),
            )
        )
        fig_s1.add_scatter(
            x=[P.price],
            y=[P.debt_fraction],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="current",
        )
        fig_s1.update_layout(
            height=360,
            title="Sensitivity to price",
            xaxis_title="price",
            yaxis_title="Debt fraction",
        )
        st.plotly_chart(fig_s1, use_container_width=True, config={"displaylogo": False})

    with c2:
        fig_s2 = go.Figure(
            go.Heatmap(
                z=sensitivity_df_pp_per_1pp,
                x=prices,
                y=dfs,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Δ APY pp per 1pp DF"),
            )
        )
        fig_s2.add_scatter(
            x=[P.price],
            y=[P.debt_fraction],
            mode="markers",
            marker=dict(size=8, color="black"),
            name="current",
        )
        fig_s2.update_layout(
            height=360,
            title="Sensitivity to debt fraction",
            xaxis_title="price",
            yaxis_title="Debt fraction",
        )
        st.plotly_chart(fig_s2, use_container_width=True, config={"displaylogo": False})

with tabs[3]:
    st.markdown(
        "**What this shows:** one-dimensional slices through the policy at the current state.  \n"
        "**Interpretation:** these plots are easier than the full surface for local sanity checks. "
        "Use them to answer questions like: what happens if price drops slightly, or if debt fraction drifts upward from here?"
    )
    c1, c2, c3 = st.columns(3)

    with c1:
        xs = np.linspace(0.95, 1.05, 301)
        ys = [apy_at_point(P, float(x), P.debt_fraction) * 100 for x in xs]
        fig1 = go.Figure()
        fig1.add_scatter(x=xs, y=ys, mode="lines", name="APY vs price")
        fig1.add_vline(x=P.price, line_dash="dot")
        fig1.update_layout(
            height=320,
            xaxis_title="price",
            yaxis_title="APY %",
            title="APY vs price",
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

    with c2:
        xs = np.linspace(0.0, 1.0, 301)
        ys = [apy_at_point(P, P.price, float(x)) * 100 for x in xs]
        fig2 = go.Figure()
        fig2.add_scatter(x=xs, y=ys, mode="lines", name="APY vs debt fraction")
        fig2.add_vline(x=P.debt_fraction, line_dash="dot")
        fig2.add_vline(x=P.target_fraction, line_dash="dot", line_color="gray")
        fig2.update_layout(
            height=320,
            xaxis_title="Debt fraction",
            yaxis_title="APY %",
            title="APY vs debt fraction",
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

    with c3:
        xs = np.linspace(0.95, 1.05, 301)
        y_price = [((P.price_peg - float(x)) * ONE) / P.sigma for x in xs]
        y_debt = np.full_like(xs, P.debt_fraction / P.target_fraction)
        fig3 = go.Figure()
        fig3.add_scatter(x=xs, y=y_price, mode="lines", name="price term")
        fig3.add_scatter(x=xs, y=y_debt, mode="lines", name="debt term")
        fig3.add_vline(x=P.price, line_dash="dot")
        fig3.update_layout(
            height=320,
            xaxis_title="price",
            yaxis_title="Term value",
            title="Price term vs debt term",
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})

with tabs[4]:
    st.markdown(
        "**What this shows:** a point-in-time breakdown of policy power into the positive price term and the negative debt term.  \n"
        "**Interpretation:** this is the cleanest single-point explanation of *why* the current rate is where it is. "
        "If the price bar dominates, depeg is driving the policy; if the debt bar dominates, debt saturation is the main force."
    )

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
    waterfall.update_layout(height=360, yaxis_title="Contribution to power")
    st.plotly_chart(waterfall, use_container_width=True, config={"displaylogo": False})