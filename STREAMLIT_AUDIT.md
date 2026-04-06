# Streamlit Deployment Audit

Date: 2026-04-06

Scope audited:
- `apps/semilog_irm_app.py`
- `apps/crvusd_monetary_policy_app.py`
- `apps/piecewise_linear_irm_app.py`
- `apps/secondary_mp_tool.py`

---

## Executive summary

All four apps are functional and already include strong model explanations, interactive controls, and charts. The biggest improvement opportunities are:

1. **Input scale and formatting consistency** (mixed decimal/percent mental model across apps).
2. **Edge-case handling and guardrails** (especially around near-zero/near-one ranges, rate explosion, and silently swallowed exceptions).
3. **Performance tuning** for dense grid computations in the crvUSD dashboard.
4. **Deployment hardening** for Streamlit Cloud stability (version pinning and deterministic plotting behavior).

---

## Cross-app findings and recommendations

### A) Scale/formatting improvements

**Observed**
- Some controls are shown in raw decimal with high precision while outputs are percentages.
- Labels are not fully standardized (`rate_min`, `r0`, `price_crvusd`, etc.).

**Recommendations**
- Present rates in **percent units in UI labels** while keeping decimal internally.
- Add small inline reminders near controls, e.g. `0.10 = 10%`.
- Standardize naming patterns:
  - Inputs: `Base Rate (%)`, `Max Rate (%)`, `Target Utilization (%)`
  - Outputs: `Borrow Rate (APY %)`
- Use consistent numeric formatting:
  - Price: `%.4f`
  - Utilization: `%.1f%%`
  - APY: `%.2f%%`

### B) Edge-case controls

**Observed**
- Piecewise and semilog apps validate parameters, but UI still allows boundary combinations that can produce extreme curves.
- SMP app catches errors in loop with a bare `except`, which hides root causes.

**Recommendations**
- Replace bare `except` with `except Exception as e` and log or surface diagnostics.
- Add explicit cap warnings:
  - “Results may saturate due to max-rate cap.”
- Add small `st.warning` when:
  - `u` is near 0 or 1
  - slope/ratio choices imply very steep derivatives
  - sigma is at extremes

### C) Performance and responsiveness

**Observed**
- The crvUSD app computes several dense surfaces using nested Python loops.

**Recommendations**
- Cache expensive computations with `@st.cache_data` for repeated grids.
- Consider reducing default grid density and expose “High resolution” toggle.
- Centralize APY-grid helper to avoid repeated recomputation for adjacent panels.

### D) Deployment reliability

**Observed**
- Repo currently includes no deployment-focused README guidance and no explicit app matrix in docs.

**Recommendations**
- Keep a single deployment matrix in `README.md` (added in this change).
- Pin library versions in `requirements.txt` (especially `streamlit`, `numpy`, `matplotlib`, `plotly`, `pandas`).
- Add lightweight smoke checks in CI:
  - `python -m compileall apps interest_rate_models`
  - import checks for each app module.

---

## Per-app audit notes

## 1) Semi-Log IRM (`apps/semilog_irm_app.py`)

**Strengths**
- Good parameter validation and graceful failure path.
- Helpful log-scale checkbox for exponential behavior.

**Improvement shortlist**
- `rate_min` slider precision (`%.12f` and tiny step) is difficult to operate manually.
- Add optional percent-mode controls to reduce confusion.
- Add “preset profiles” (conservative/aggressive) for easier scenario testing.

## 2) crvUSD Monetary Policy (`apps/crvusd_monetary_policy_app.py`)

**Strengths**
- Rich explanatory content and multiple visual diagnostics.
- Captures contract-like behavior, including cap and edge-case semantics.

**Improvement shortlist**
- Use caching for 3D/heatmap/slice calculations.
- Add explicit on-chart marker/label when the cap is active.
- Consider smaller default grid with user-selectable “quality mode.”

## 3) Piecewise Linear IRM (`apps/piecewise_linear_irm_app.py`)

**Strengths**
- Clear formula and continuity-oriented visualization.
- Dynamic lower bound for `r2 >= r1` is a good guardrail.

**Improvement shortlist**
- Add optional log-scale Y (useful when `r2` is large).
- Add explicit continuity metric around kink (`left-right diff`).
- Add presets for common kink/slope profiles.

## 4) Secondary Monetary Policy (`apps/secondary_mp_tool.py`)

**Strengths**
- Good introductory explanation and intuitive parameterization.
- Useful single-point + curve view.

**Improvement shortlist**
- Replace bare `except` in curve loop to avoid masking faults.
- Add validator feedback near control panel for invalid combinations.
- Add export/download for params JSON to match other apps.

---

## Suggested implementation order

1. **Quick wins (low risk, immediate UX impact)**
   - Standardize labels/formatting to percent-friendly language.
   - Replace bare exception handling.
   - Add edge-case warnings near extreme values.

2. **Performance pass**
   - Cache all grid computations in crvUSD app.
   - Add quality/resolution toggle.

3. **Reliability pass**
   - Pin dependencies.
   - Add smoke checks.

4. **Product polish**
   - Add presets and JSON export consistency across all apps.

