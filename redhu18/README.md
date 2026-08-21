# redhu18 — Hywind Tampen submission & live dashboard

- `submission.csv` — final competition forecast for HYT-HY09 (+30/+60 min,
  point + q0.05/q0.95 calibrated bands), 694 rows per the official format.
- `hywind-app/` — self-contained operator dashboard (Dash). Run:
  `pip install -r requirements.txt && gunicorn -w 2 -b 0.0.0.0:8506 app:server`
  or `docker build -t hywind-app . && docker run -p 8506:8506 hywind-app`.

Model: event-weighted LightGBM+XGBoost ensemble (3-seed bagged) on causal
sensor features + far-field ERA5 with travel-time physics; conformal-
calibrated 90% bands; validated on two out-of-time holdouts (−33…−50% RMSE
vs persistence).
