# superRR-selector-S2

Dashboard for **SupeRR Selector Season 2 – Round 1**: who are the most reliable batters and bowlers under pressure?

## Method (short)
1. Win-probability model per innings (out-of-fold).
2. Leverage index per delivery. Pressure = top 25% leverage.
3. Each delivery gets a z-score (outcome vs expectation).
4. Players are rated on pressure level, lift vs their usual level, and consistency, with small samples shrunk.
5. Final rank = median rank over 100 bootstrap resamples.

Data: IPL + T20I, 2022–26 (818 matches, 195,124 deliveries).

## Run
pip install -r requirements.txt
streamlit run app.py

## Files
- `app.py` – Streamlit dashboard
- `dashboard_data/` – ratings and chart data used by the app
