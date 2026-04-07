# IRM Streamlit Deployments

This repository contains Streamlit apps for interest-rate and monetary-policy simulations.

## Live deployments

- **Semi-Log IRM App**: https://semilogirmapp.streamlit.app/
- **crvUSD Monetary Policy App**: https://crvusd-monetary-policy.streamlit.app/
- **Piecewise Linear IRM App**: https://piecewiselinearirm.streamlit.app/
- **Secondary Monetary Policy App**: https://secondarymp.streamlit.app/

## Local development

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run each app

```bash
streamlit run apps/semilog_irm_app.py
streamlit run apps/crvusd_monetary_policy_app.py
streamlit run apps/piecewise_linear_irm_app.py
streamlit run apps/secondary_mp_tool.py
```

## Repo structure

- `apps/`: Streamlit frontends.
- `interest_rate_models/`: Core model logic used by apps.
- `STREAMLIT_AUDIT.md`: Deployment and UX/edge-case audit with recommended improvements.
