# Save as: crvusd_policy_v4.py
# Run: streamlit run crvusd_policy_v4.py

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Constants
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
ONE = 10**18  # on-chain fixed-point scale

# Contract-aligned constants
MAX_RATE_RAW = 43_959_106_799  # per-second, 1e18 scaled (~300% APY cap in AggMonetaryPolicy)
TARGET_REMAINDER = 10**17      # -> T = 0.1 in 1e18 scale for the ceiling multiplier
T = TARGET_REMAINDER / 1e18

# ──────────────────────────────────────────────────────────────────────────────
# Core model (contract-aligned core + extras)
@dataclass
class PolicyInputs:
    rate0_apy: float
    sigma: float            # 1e18-scaled (e.g., 7e15), MUST be in [1e14, 1e18]
    target_fraction: float  # 0..1 (strictly >0 onchain)
    pegkeeper_debt: float
    total_debt: float
    price: float            # $ (oracle aggregate)
    price_peg: float = 1.0  # typically 1.00

    # Extras we expose via UI
    extra_const_apy: float = 0.0  # added after the exp() core, like a floor/shift
    ceiling_util: float = 0.0     # f in [0, 0.99], per-market debt/ceiling utilization

    @property
    def debt_fraction(self) -> float:
        return float(np.clip(self.pegkeeper_debt / max(self.total_debt, 1.0), 0.0, 1.0))

    # 1e18-scaled helpers
    @property
    def price_raw(self) -> int: return int(self.price * ONE)
    @property
    def price_peg_raw(self) -> int: return int(self.price_peg * ONE)
    @property
    def df_raw(self) -> int: return int(self.debt_fraction * ONE)
    @property
    def target_fraction_raw(self) -> int: return int(self.target_fraction * ONE)

    # rate0 as per-second decimal and 1e18 fp
    @property
    def rate0_per_sec(self) -> float:
        return (1.0 + self.rate0_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0
    @property
    def rate0_raw(self) -> float:
        return self.rate0_per_sec * ONE

    # extra_const in per-second (decimal) and 1e18 fp
    @property
    def extra_per_sec(self) -> float:
        return (1.0 + self.extra_const_apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0
    @property
    def extra_raw(self) -> float:
        return self.extra_per_sec * ONE


def compute_power(price_peg_raw: int, price_raw: int, df_raw: int, sigma: float, target_fraction_raw: int) -> float:
    """power = ((price_peg_raw - price_raw) * 1e18)/sigma - (df_raw * 1e18)/target_fraction_raw (1e18-scaled)"""
    return ((price_peg_raw - price_raw) * ONE) / sigma - (df_raw * ONE) / target_fraction_raw

def per_second_rate_core(rate0_raw: float, power: float) -> float:
    """Core per-second decimal: rate0 * exp(power/1e18), before extras."""
    return (rate0_raw / ONE) * math.exp(power / ONE)

def apply_contract_extras(core_per_sec: float, extra_raw: float, ceiling_util: float) -> float:
    """
    Mirrors AggMonetaryPolicy post-processing:
      1) Add extra_const (per-second 1e18-scaled) after the exponential core
      2) Multiply by the debt-ceiling utilization multiplier
      3) Cap final per-second (1e18) at MAX_RATE_RAW
    """
    # Add extra_const (convert to decimal)
    rate_per_sec = core_per_sec + (extra_raw / ONE)

    # Debt-ceiling multiplier: (1 - T) + T/(1 - f), clamp near f=1.0
    eps = 1e-12
    f = float(np.clip(ceiling_util, 0.0, 0.999999))
    mult = (1.0 - T) + T / max(1.0 - f, eps)
    rate_per_sec *= mult

    # Cap to MAX_RATE (cap applies to 1e18 per-second)
    rate_raw_capped = min(rate_per_sec * ONE, MAX_RATE_RAW)
    return rate_raw_capped / ONE  # back to decimal per-second


def annualize(rate_per_sec: float) -> float:
    return (1.0 + rate_per_sec) ** SECONDS_PER_YEAR - 1.0

def deannualize(apy: float) -> float:
    return (1.0 + apy) ** (1.0 / SECONDS_PER_YEAR) - 1.0

# Convenience: full stack computation at arbitrary (price, df)
def apy_at_point_full(P: PolicyInputs, price: float, df: float) -> float:
    # Contract edge case: if pk_debt > 0 and total_debt == 0 → return 0
    if P.pegkeeper_debt > 0 and P.total_debt == 0:
        return 0.0
    price_raw = int(price * ONE)
    df_raw = int(np.clip(df, 0.0, 1.0) * ONE)
    pow_ = compute_power(P.price_peg_raw, price_raw, df_raw, P.sigma, P.target_fraction_raw)
    core_ps = per_second_rate_core(P.rate0_raw, pow_)
    final_ps = apply_contract_extras(core_ps, P.extra_raw, P.ceiling_util)
    return annualize(final_ps)

# ──────────────────────────────────────────────────────────────────────────────
# 2D dashboard helpers (shared)
def apy_grid_full(P: PolicyInputs, prices: np.ndarray, dfs: np.ndarray):
    PX, DF = np.meshgrid(prices, dfs)
    Z = np.zeros_like(PX)
    for i in range(DF.shape[0]):
        for j in range(PX.shape[1]):
            Z[i, j] = apy_at_point_full(P, PX[i, j], DF[i, j])
    return PX, DF, Z

def neutrality_curve_df(P: PolicyInputs, prices: np.ndarray) -> np.ndarray:
    # power=0 ⇒ DF = Target * (peg - price) * (1e18/sigma)
    df = P.target_fraction * np.maximum(0.0, (P.price_peg - prices)) * (ONE / P.sigma)
    return np.clip(df, 0.0, 1.0)

# ──────────────────────────────────────────────────────────────────────────────
# App
st.set_page_config(page_title="crvUSD Monetary Policy", layout="wide")
st.title("🏦 crvUSD Monetary Policy — System Model")

# Intro: formula + quick explanation (APY-first terminology)
st.markdown(
    r"""
**Policy mechanics**  
Per contract logic, the per-second borrow rate is:
$$
\underbrace{\text{rate}}_{\text{per-second}} \;=\;
\underbrace{\text{rate0}\cdot e^{\text{APY pressure}}}_{\text{core}} \;
+\; \underbrace{\text{extra\_const}}_{\text{added after core}}
$$
then multiplied by a **debt-ceiling utilization** factor and finally **capped**:
$$
\text{rate} \leftarrow
\min\!\Big( \big(\text{rate0}\cdot e^{\text{APY pressure}} + \text{extra\_const}\big)\cdot \text{mult}(f) ,\; \text{MAX\_RATE}\Big)
$$
where the **APY pressure** (a.k.a. *power* in code) is
$$
\text{APY pressure} = \frac{\text{price}_{peg} - \text{price}_{crvusd}}{\sigma}
               - \frac{\text{DebtFraction}}{\text{TargetFraction}},\qquad
\text{DebtFraction}=\frac{\text{PegKeeperDebt}}{\text{TotalDebt}}
$$
and
$$
\text{mult}(f) \approx (1 - 0.1) + \frac{0.1}{1 - f},\quad f=\frac{\text{debt\_for}}{\text{debt\_ceiling}}\in[0,1) \;.
$$
**Annualized APY** shown here is $ (1 + \text{rate})^{\text{seconds per year}} - 1 $.  
**Edge case** per contract: if **PegKeeperDebt > 0** and **TotalDebt = 0**, the rate returns **0**.

**Intuition** 
- **Price ↓ ⇒ APY ↑** (below-peg raises APY pressure).
- **Price ↑ ⇒ APY ↓** (above-peg lowers APY pressure).
- **DebtFraction ↓ ⇒ APY ↑** (below-target DF raises APY pressure).
- **DebtFraction ↑ ⇒ APY ↓** (above-target DF lowers APY pressure).
"""
)

# Sidebar inputs
st.sidebar.header("Inputs")

# Defaults from deployed contract
_default_rate0_raw = 3_488_077_118  # per-second, 1e18-scaled
_default_rate0_ps  = _default_rate0_raw / ONE
_default_rate0_apy = (1.0 + _default_rate0_ps) ** SECONDS_PER_YEAR - 1.0

rate0_apy = st.sidebar.slider(
    "Baseline rate0 (APY)",
    0.0, 3.0, float(_default_rate0_apy), 0.001, format="%.3f",
    help="Baseline APY when APY pressure = 0. Contract cap ≈ 300% APY."
)

# sigma control is log10(σ) in UI; convert internally (auto-bounds to [1e14,1e18])
_default_log_sigma = float(np.log10(7e15))
log_sigma_ui = st.sidebar.slider(
    "Price sensitivity (σ) — log10 scale",
    14.0, 18.0, _default_log_sigma, 0.001,
    help="This control is log10(σ). Example: σ=7e15 ⇒ log10(σ)=15.845. Smaller σ = steeper APY response."
)
sigma = float(np.clip(10 ** log_sigma_ui, 1e14, 1e18))

target_fraction = st.sidebar.slider(
    "Utilization target (TargetFraction)",
    0.01, 1.0, 0.20, 0.01,
    help="Target PegKeeper debt share. Higher Target weakens the DF penalty in the APY core."
)

# Debt controls: single DF
st.sidebar.subheader("Debt")
debt_fraction_ui = st.sidebar.slider(
    "PegKeeper utilization (DebtFraction)", 0.0, 1.0, 0.05, 0.005,
    help="Policy uses only this ratio in the APY core."
)

# Internally derive absolute values (kept for compatibility with the model structure)
total_debt = 1.0
pegkeeper_debt = float(debt_fraction_ui) * total_debt

st.sidebar.subheader("Price")
price = st.sidebar.slider(
    "price_crvusd", 0.90, 1.10, 1.00, 0.0001, format="%.4f",
    help="Below peg ⇒ APY pressure ↑ ⇒ APY ↑. Above peg ⇒ APY ↓."
)
price_peg = st.sidebar.number_input(
    "price_peg (target)", 0.90, 1.10, 1.00, 0.0001, format="%.4f",
    help="Target price, typically 1.00."
)

st.sidebar.subheader("Contract extras")
extra_const_apy = st.sidebar.slider(
    "extra_const (APY add-on)",
    0.0, 3.0, 0.00, 0.001, format="%.3f",
    help="Added after the exponential core. Shifts the entire APY curve upward."
)
ceiling_util = st.sidebar.slider(
    "Debt-ceiling utilization f",
    0.0, 0.99, 0.00, 0.01,
    help="Per-market stress knob. Multiplier ≈ 0.9 + 0.1/(1−f). Final APY still capped."
)

log_y = st.sidebar.checkbox("Log-scale Y for 2D slices", value=False)

# Build inputs & compute outputs
P = PolicyInputs(
    rate0_apy=rate0_apy,
    sigma=sigma,
    target_fraction=target_fraction,
    pegkeeper_debt=pegkeeper_debt,
    total_debt=total_debt,
    price=price,
    price_peg=price_peg,
    extra_const_apy=extra_const_apy,
    ceiling_util=ceiling_util,
)

# Current point computation
if P.pegkeeper_debt > 0 and P.total_debt == 0:
    annual_rate = 0.0
    power = 0.0
    rate_per_sec_core = 0.0
    rate_per_sec = 0.0
else:
    power = compute_power(P.price_peg_raw, P.price_raw, P.df_raw, P.sigma, P.target_fraction_raw)
    rate_per_sec_core = per_second_rate_core(P.rate0_raw, power)
    rate_per_sec = apply_contract_extras(rate_per_sec_core, P.extra_raw, P.ceiling_util)
    annual_rate = annualize(rate_per_sec)

rate_raw = min(rate_per_sec * ONE, MAX_RATE_RAW)

# Top metrics
c1, c2, c3 = st.columns(3)
c1.metric("DebtFraction", f"{P.debt_fraction:.2%}")
c2.metric("Annualized (APY)", f"{annual_rate:.2%}")
c3.metric("Final rate (per-sec · 1e18)", f"{rate_raw:,.0f}")

# System inputs snapshot + micro-sensitivity chips
colA, colB = st.columns([2,1])
with colA:
    st.subheader("System Inputs")
    inputs_df = pd.DataFrame(
        {"value":[f"{P.rate0_apy:.3f}", f"{P.sigma:.3e}", f"{P.target_fraction:.2%}",
                   f"{P.pegkeeper_debt:,.0f}", f"{P.total_debt:,.0f}", f"{P.debt_fraction:.2%}",
                   f"{P.price_peg:.4f}", f"{P.price:.4f}",
                   f"{P.extra_const_apy:.3f}", f"{P.ceiling_util:.2f}"]},
        index=["rate0_apy","sigma","TargetFraction","PegKeeperDebt","TotalDebt","DebtFraction",
               "price_peg","price_crvusd","extra_const_apy","ceiling_util(f)"],
    )
    st.table(inputs_df)
with colB:
    st.subheader("Live Sensitivity")
    dprice = 0.001
    ddf = 0.01
    apy_now  = apy_at_point_full(P, P.price, P.debt_fraction)
    apy_pup  = apy_at_point_full(P, P.price + dprice, P.debt_fraction)
    apy_pdn  = apy_at_point_full(P, P.price - dprice, P.debt_fraction)
    apy_dfup = apy_at_point_full(P, P.price, P.debt_fraction + ddf)
    apy_dfdn = apy_at_point_full(P, P.price, P.debt_fraction - ddf)
    bps = lambda x: f"{(x*1e4):+.1f} bps"
    st.metric("+0.001 price →", bps(apy_pup-apy_now))
    st.metric("-0.001 price →", bps(apy_pdn-apy_now))
    st.metric("+1pp DF →",     bps(apy_dfup-apy_now))
    st.metric("-1pp DF →",     bps(apy_dfdn-apy_now))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Explainer (one paragraph)
st.markdown(
    r"""
**Reading the visuals.** Price below peg and/or DebtFraction below Target **raises** the **APY pressure** (core) ⇒ **raises** APY; the opposite **lowers** it.  
The **extra_const** shifts the curve up; the **debt-ceiling multiplier** increases APY as a market approaches its ceiling; and the final per-second rate is capped at ~**300% APY** equivalent.
"""
)

# ──────────────────────────────────────────────────────────────────────────────
# Interactive 3D Surface (Plotly) — contract-faithful
st.subheader("Interactive 3D surface — APY(price, DebtFraction)")

# Grid
px = np.linspace(0.90, 1.10, 121)
DF = np.linspace(0.0, 1.0, 101)
PX, DFg = np.meshgrid(px, DF)

# Compute APY over grid
Z = np.zeros_like(PX)
for i in range(DFg.shape[0]):
    for j in range(PX.shape[1]):
        Z[i, j] = apy_at_point_full(P, PX[i, j], DFg[i, j])

# Color by APY ÷ baseline (normalized 0.5×..1.5×)
Zbase = max(P.rate0_apy, 1e-12)
Zratio = Z / Zbase
Zratio_norm = (np.clip(Zratio, 0.5, 1.5) - 0.5) / 1.0  # → [0,1]

fig = go.Figure()
div_colorscale = [
    [0.00, "#2c7bb6"],
    [0.45, "#abd9e9"],
    [0.50, "#ffffbf"],
    [0.55, "#fdae61"],
    [1.00, "#d7191c"],
]
fig.add_surface(
    x=PX, y=DFg, z=Z,
    surfacecolor=Zratio_norm,
    colorscale=div_colorscale, cmin=0.0, cmax=1.0,
    colorbar=dict(
        title="APY ÷ baseline",
        tickmode="array", tickvals=[0.0, 0.5, 1.0], ticktext=["0.5×", "1.0×", "1.5×"],
        thickness=10, len=0.55, x=1.02, y=0.5,
    ),
    hovertemplate=(
        "price=%{x:.4f}<br>"
        "DebtFraction=%{y:.2%}<br>"
        "APY=%{z:.2%}<extra></extra>"
    ),
    showscale=True,
)
# Crosshair slices
DF_line = np.linspace(0, 1, 120)
Z_line1 = [apy_at_point_full(P, P.price, d) for d in DF_line]
fig.add_scatter3d(x=[P.price]*len(DF_line), y=DF_line, z=Z_line1, mode="lines", name="slice @ price", line=dict(width=6))

PX_line = np.linspace(0.95, 1.05, 160)
Z_line2 = [apy_at_point_full(P, p, P.debt_fraction) for p in PX_line]
fig.add_scatter3d(x=PX_line, y=[P.debt_fraction]*len(PX_line), z=Z_line2, mode="lines", name="slice @ DF", line=dict(width=6))

# Current point marker
fig.add_scatter3d(x=[P.price], y=[P.debt_fraction], z=[annual_rate], mode="markers", marker=dict(size=6), name="current")

fig.update_layout(
    scene=dict(
        xaxis_title="price_crvusd",
        yaxis_title="DebtFraction",
        zaxis_title="APY",
        xaxis=dict(range=[px.min(), px.max()]),
        yaxis=dict(range=[0, 1]),
    ),
    legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0.0, bgcolor="rgba(255,255,255,0.6)"),
    margin=dict(l=0, r=60, t=30, b=70),
    height=640,
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 2D Policy Dashboard (live) — contract-faithful computations
st.markdown("### 🔎 2D Policy View")

# Row A
rA1, rA2, rA3 = st.columns([1,1,1])

with rA1:
    st.markdown("#### APY vs **Price** (holding DebtFraction fixed)")
    xs = np.linspace(0.95, 1.05, 301)
    fig1 = go.Figure()
    df_lines = (0.0, 0.05, 0.10, 0.20, 0.40, 0.60)
    for df in df_lines:
        ys = [apy_at_point_full(P, x, df) for x in xs]
        fig1.add_scatter(x=xs, y=ys, mode="lines", name=f"DF {df:.0%}",
                         line=dict(width=2 if abs(df-P.debt_fraction) < 1e-6 else 1))
    cur_apy = apy_at_point_full(P, P.price, P.debt_fraction)
    fig1.add_vline(x=P.price, line_dash="dot")
    fig1.add_hline(y=P.rate0_apy, line_dash="dot", line_color="gray")
    fig1.add_scatter(x=[P.price], y=[cur_apy], mode="markers", name="current", marker=dict(size=8))
    fig1.update_layout(xaxis_title="price_crvusd", yaxis_title="APY",
                       height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    if log_y: fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})
    st.caption("Price effect enters the APY core as (peg − price)/σ. Smaller σ ⇒ steeper APY response. Curves include extra_const, ceiling multiplier, and cap.")

with rA2:
    st.markdown("#### APY vs **DebtFraction** (holding Price fixed)")
    xs = np.linspace(0.0, 1.0, 301)
    fig2 = go.Figure()
    # Show TARGET variants at price ≈ 1.0 (more useful operationally)
    target_lines = (max(P.target_fraction*0.5, 0.01), P.target_fraction, min(P.target_fraction*1.5, 1.0))
    labels = ["Target ×0.5", "Target (cur)", "Target ×1.5"]
    for tgt, lab in zip(target_lines, labels):
        P_tmp = PolicyInputs(
            rate0_apy=P.rate0_apy, sigma=P.sigma, target_fraction=float(tgt),
            pegkeeper_debt=P.pegkeeper_debt, total_debt=P.total_debt,
            price=1.00, price_peg=P.price_peg,
            extra_const_apy=P.extra_const_apy, ceiling_util=P.ceiling_util,
        )
        ys = [apy_at_point_full(P_tmp, P_tmp.price, df) for df in xs]
        width = 3 if np.isclose(tgt, P.target_fraction) else 1
        fig2.add_scatter(x=xs, y=ys, mode="lines", name=lab, line=dict(width=width))
    cur_apy = apy_at_point_full(P, P.price, P.debt_fraction)
    fig2.add_vline(x=P.debt_fraction, line_dash="dot")
    fig2.add_vline(x=P.target_fraction, line_dash="dot", line_color="gray")
    fig2.add_hline(y=P.rate0_apy, line_dash="dot", line_color="gray")
    fig2.add_scatter(x=[P.debt_fraction], y=[cur_apy], mode="markers", name="current", marker=dict(size=8))
    fig2.update_layout(xaxis_title="DebtFraction", yaxis_title="APY",
                       height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    if log_y: fig2.update_yaxes(type="log")
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
    st.caption("Utilization effect enters as −DF/Target. Panel compares different Target settings at price≈1.0. Curves include extras and cap.")

with rA3:
    st.markdown("#### Contour — **APY** over (Price, DebtFraction) with Baseline Frontier")
    prices = np.linspace(0.95, 1.05, 181)
    dfs    = np.linspace(0.0, 1.0, 121)
    PXc, DFc, Zc = apy_grid_full(P, prices, dfs)
    mask = np.abs(Zc - P.rate0_apy) < (50/1e4)  # 50 bps neutral band
    df_curve = neutrality_curve_df(P, prices)
    fig3 = go.Figure()
    fig3.add_trace(go.Heatmap(z=Zc, x=prices, y=dfs, colorscale="Viridis", colorbar=dict(title="APY")))
    fig3.add_trace(go.Contour(z=mask.astype(int), x=prices, y=dfs, showscale=False, contours_coloring='lines',
                              line=dict(color="white")))
    fig3.add_scatter(x=prices, y=df_curve, mode="lines", name="baseline-APY frontier (core)", line=dict(color="white", width=3, dash="dash"))
    fig3.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", name="current", marker=dict(size=9, color="red"))
    fig3.update_layout(xaxis_title="price_crvusd", yaxis_title="DebtFraction",
                       height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})
    st.caption("Heatmap shows final APY (incl. extras & cap). Dashed line marks where the core implies APY ≈ rate0.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Row B
rB1, rB2, rB3 = st.columns([1,1,1])

with rB1:
    # Single policy map with toggle preserved, but APY-first wording
    st.markdown("#### Policy map — **(peg − price, DF / Target)**")
    map_mode = st.radio("Metric", ["APY pressure (core)", "APY (final)"], horizontal=True, key="policy_map_mode")
    prices = np.linspace(0.95, 1.05, 181)
    dfs    = np.linspace(0.0, 1.0, 121)
    PXm, DFm, Zm_final = apy_grid_full(P, prices, dfs)
    dev_x   = P.price_peg - prices
    ratio_y = dfs / max(P.target_fraction, 1e-12)

    if map_mode == "APY pressure (core)":
        Z_show = np.zeros_like(PXm)
        for i in range(DFm.shape[0]):
            for j in range(PXm.shape[1]):
                Z_show[i, j] = compute_power(P.price_peg_raw, int(PXm[i, j]*ONE), int(DFm[i, j]*ONE), P.sigma, P.target_fraction_raw) / ONE
        colorscale = "RdBu"
        reversescale = True
        cbar_title = "APY pressure (core index)"
    else:
        Z_show = Zm_final
        colorscale = "Viridis"
        reversescale = False
        cbar_title = "APY"

    # Baseline frontier (core)
    line_y = (ONE / P.sigma) * (P.price_peg - prices)

    fig4 = go.Figure(data=go.Heatmap(
        z=Z_show, x=dev_x, y=ratio_y, colorscale=colorscale, reversescale=reversescale,
        colorbar=dict(title=cbar_title)
    ))
    fig4.add_scatter(x=dev_x, y=line_y/max(P.target_fraction,1e-12), mode="lines",
                     name="baseline-APY frontier", line=dict(color="black", dash="dash"))
    fig4.add_vline(x=P.price_peg - P.price, line_dash="dot")
    fig4.add_hline(y=P.debt_fraction/max(P.target_fraction,1e-12), line_dash="dot")
    fig4.update_layout(xaxis_title="peg − price", yaxis_title="DF / Target", height=320,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig4, use_container_width=True, config={"displaylogo": False})
    st.caption("X=0 ⇒ at peg; Y=1 ⇒ at target. Toggle between **APY pressure (core)** and **final APY**. The dashed line is the baseline-APY frontier (APY ≈ rate0).")

with rB2:
    st.markdown("#### **Iso-APY** targeting curve — implied DF across price (core guidance)")
    xs = np.linspace(0.95, 1.05, 401)
    target_apy = float(max(P.rate0_apy, 0.12))  # sensible default
    rps = deannualize(target_apy)
    r_raw = rps * ONE
    power_t = ONE * math.log(max(r_raw, 1e-24) / max(P.rate0_raw, 1e-24))
    df_line = P.target_fraction * ((P.price_peg - xs) * (ONE / P.sigma) - (power_t / ONE))
    df_line = np.clip(df_line, 0.0, 1.0)
    # ±50 bps guidance band (core)
    up = target_apy + 50/1e4
    dn = max(target_apy - 50/1e4, 0.0)
    def df_for_apy(apy_val):
        rps_ = deannualize(apy_val)
        pow_ = ONE * math.log(max(rps_*ONE,1e-24) / max(P.rate0_raw,1e-24))
        out = P.target_fraction * ((P.price_peg - xs) * (ONE / P.sigma) - (pow_ / ONE))
        return np.clip(out, 0.0, 1.0)
    df_up = df_for_apy(up); df_dn = df_for_apy(dn)
    fig5 = go.Figure()
    fig5.add_scatter(x=xs, y=df_dn, mode="lines", line=dict(width=0.5), showlegend=False)
    fig5.add_scatter(x=xs, y=df_up, mode="lines", fill="tonexty", name=f"±50 bps band (core)")
    fig5.add_scatter(x=xs, y=df_line, mode="lines", name=f"APY target {target_apy:.2%} (core)", line=dict(width=3))
    fig5.add_scatter(x=[P.price], y=[P.debt_fraction], mode="markers", name="current", marker=dict(size=8))
    fig5.update_layout(xaxis_title="price_crvusd", yaxis_title="DebtFraction",
                       height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig5, use_container_width=True, config={"displaylogo": False})
    st.caption("Core guidance: DF required across price to maintain a chosen APY. Final APY will shift with extras, ceiling multiplier, and the cap.")

with rB3:
    st.markdown("#### Decomposition — `rate0` → **APY** (core → extras → ceiling → cap)")
    pp = ((P.price_peg_raw - P.price_raw) * ONE) / P.sigma
    pd = - (P.df_raw * ONE) / P.target_fraction_raw
    mult_price = math.exp(pp/ONE)
    mult_debt  = math.exp(pd/ONE)
    base_ps = P.rate0_per_sec
    step1   = base_ps * mult_price
    step2   = step1   * mult_debt
    step3   = apply_contract_extras(step2, P.extra_raw, P.ceiling_util)

    r0_apy  = (1.0 + base_ps) ** SECONDS_PER_YEAR - 1.0
    s1_apy  = (1.0 + step1  ) ** SECONDS_PER_YEAR - 1.0
    s2_apy  = (1.0 + step2  ) ** SECONDS_PER_YEAR - 1.0
    s3_apy  = (1.0 + step3  ) ** SECONDS_PER_YEAR - 1.0

    wf = go.Figure(go.Waterfall(
        name="APY",
        orientation="v",
        measure=["absolute","relative","relative","relative"],
        x=["rate0","price effect","utilization effect","extras+ceiling+cap"],
        text=[f"{r0_apy:.2%}", f"×{mult_price:.2f}", f"×{mult_debt:.2f}", ""],
        y=[r0_apy, s1_apy - r0_apy, s2_apy - s1_apy, s3_apy - s2_apy],
    ))
    wf.update_layout(showlegend=False, height=320, yaxis_title="APY")
    st.plotly_chart(wf, use_container_width=True, config={"displaylogo": False})
    st.caption("Multiplicative core steps first (price & utilization), then extras + ceiling multiplier, then the cap to reach displayed APY.")

st.markdown("---")

with st.expander("📘 Parameterization guide — how to analyze & tune this model"):
    st.markdown(r"""
**Quick mental model**

$$
\text{APY} \approx \Big(\underbrace{\text{rate0}\cdot e^{\text{APY pressure}}}_{\text{core}} + \underbrace{\text{extra}}_{\text{add-on}}\Big)\times \underbrace{\text{mult}(f)}_{\text{ceiling}}\quad\text{capped}
$$
$$
\text{APY pressure}=\frac{\text{peg}-\text{price}}{\sigma}-\frac{\text{DF}}{\text{Target}},\quad \text{DF}=\frac{\text{PegKeeperDebt}}{\text{TotalDebt}}
$$
$$
\text{mult}(f)\approx (1-0.1)+\frac{0.1}{1-f},\;f\in[0,1)
$$

**What each control does (and how to use it)**

- **rate0 (APY)** – Baseline when APY pressure = 0.  
  *Use:* lifts/lowers the whole surface uniformly. Contract default ≈ **12%** APY (from `rate0_raw=3,488,077,118`). Cap ≈ **300%** APY.

- **sigma** – Price sensitivity (slider is **log10(σ)**; the app converts to raw σ).  
  *Effect:* smaller σ ⇒ **steeper** APY response; larger σ ⇒ **flatter**.

- **TargetFraction** – Strength of the utilization term (pressure includes **− DF / Target**).  
  *Use:* **Decrease** Target to penalize high DF more (lower APY for given DF); **Increase** Target for the opposite.

- **DebtFraction (DF)** – Scenario input. Higher DF ⇒ lower APY (given Target).  
- **price / peg** – Scenario inputs. Below-peg ⇒ higher APY; above-peg ⇒ lower APY.
- **extra_const (APY add-on)** – Post-core uplift (cap still applies).
- **ceiling utilization (f)** – Multiplier that grows as a market fills; capped.

**How to analyze**

1. Start from defaults (rate0 ≈12% APY, σ=7e15, Target=0.20, price=1.00, DF≈0.05).  
2. Set **DF** and **price** to your scenario.  
3. Use **APY vs Price** to tune **σ** (steepness).  
4. Use **APY vs DF** with Target variants to tune **TargetFraction**.  
5. Inspect the **Contour** for the baseline-APY frontier (APY ≈ rate0).  
6. Use the **Waterfall** to see extras/ceiling contributions and whether you’re near the cap.  
7. For uniform shifts, adjust **rate0** (or small **extra_const** if post-core uplift is intended).

**Bounds & guardrails:** on-chain **1e14 ≤ σ ≤ 1e18**, **0 < Target ≤ 1**, **rate ≤ cap**.
""")
