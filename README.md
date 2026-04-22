# PNW Regional Economic Health Dashboard

![Phase 1 - Data Pipeline](https://img.shields.io/badge/Phase%201-Complete-brightgreen?style=flat-square)
![Phase 2 - Dashboard](https://img.shields.io/badge/Phase%202-Complete-brightgreen?style=flat-square)
![Phase 3 - DS Layer](https://img.shields.io/badge/Phase%203-Complete-brightgreen?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-50%20passing-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)

An interactive dashboard tracking key economic indicators for Portland and the Pacific Northwest. Built as a portfolio project demonstrating data engineering, data science, and business communication skills.

**Live app:** https://pnw-econ-health-tracker.streamlit.app
**Tech stack:** Python · FRED API · Pandas · Prophet · Plotly · Streamlit

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pipeline (FRED ingestion, normalization, CSV caching) | ✅ Complete |
| 2 | Core Streamlit dashboard (visualizations, interactivity, deployment) | ✅ Complete |
| 3 | DS layer (Prophet forecasting, anomaly detection, "So What?" narrative) | ✅ Complete |

---

## Indicators Tracked

| Indicator | Source | Series ID | Frequency |
|-----------|--------|-----------|-----------|
| Oregon Unemployment Rate | FRED | `ORUR` | Monthly |
| CPI – Western Urban Areas | FRED | `CUURA400SA0` | Monthly |
| Portland Metro Home Price Index | FRED | `ATNHPIUS38900Q` | Quarterly |
| Federal Funds Rate | FRED | `FEDFUNDS` | Monthly |

---

## Setup

### 1. Clone and create virtual environment
```bash
git clone <your-repo-url>
cd pnw_dashboard
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** `prophet` requires `cmdstanpy` and its Stan compiler backend.
> On first install, run the following once to download the compiler:
> ```python
> import cmdstanpy; cmdstanpy.install_cmdstan()
> ```
> This is handled automatically on Streamlit Community Cloud via `packages.txt`.

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your keys:
#   FRED_API_KEY — free at https://fred.stlouisfed.org/docs/api/api_key.html
#   BLS_API_KEY  — optional, free at https://www.bls.gov/developers/
```

### 3. Run the data pipeline
```bash
python run_pipeline.py
# Force a fresh pull (ignore cache):
python run_pipeline.py --no-cache
# Custom date range:
python run_pipeline.py --start 2010-01-01 --end 2024-12-31
```

Outputs:
- `data/raw/<key>.csv` — individual series CSVs
- `data/processed/all_indicators.csv` — merged, ready for the dashboard

### 4. Run the dashboard locally
```bash
streamlit run app.py
```

The app loads from `data/processed/all_indicators.csv` if it exists (fast path),
otherwise it fetches live from the FRED API. Use the **Refresh from FRED** button
in the sidebar to force a fresh pull at any time.

Prophet forecasts and anomaly detection run on first load and are cached for
one hour. Forecasts may take 10–20 seconds to compute on a cold start.

### 5. Run tests
```bash
pytest tests/ -v
```

---

## Project Structure

```
pnw_dashboard/
├── app.py                     # Streamlit entry point
├── run_pipeline.py            # Phase 1 pipeline entry point
├── requirements.txt
├── packages.txt               # System dependencies for Streamlit Cloud (libgomp1)
├── .env.example
├── .gitignore
├── src/
│   ├── config.py              # Series IDs, labels, Phase 3 constants
│   ├── data_fetcher.py        # FRED + BLS ingestion, caching
│   ├── chart_helpers.py       # Plotly chart builders, forecast overlay, anomaly markers
│   ├── forecaster.py          # Prophet-based 12-month forecasting (Phase 3)
│   ├── anomaly_detector.py    # Rolling z-score anomaly detection (Phase 3)
│   └── narrative.py           # Rule-based plain-language narrative generation (Phase 3)
├── data/
│   ├── raw/                   # Cached individual series CSVs (gitignored)
│   └── processed/             # Merged output for dashboard (gitignored)
└── tests/
    ├── test_data_fetcher.py
    ├── test_forecaster.py
    ├── test_anomaly_detector.py
    └── test_narrative.py
```

---

## Phase 1 Summary — Data Pipeline

*Completed March 2026.*

| | |
|---|---|
| **Series ingested** | 4 (unemployment, CPI, home prices, fed funds rate) |
| **Date range** | 2000-01-01 → present |
| **Processed output** | `data/processed/all_indicators.csv` (314 rows × 4 cols) |
| **Unit tests** | 17 passing — normalisation, caching, key validation |
| **Pipeline runtime** | ~1 second |

The pipeline pulls, normalises, and caches all configured FRED series. Series with
different frequencies (monthly vs quarterly) are outer-joined — sparse values in
quarterly columns are expected and handled by the dashboard layer.

---

## Phase 2 Summary — Core Dashboard

*Completed April 2026.*

| | |
|---|---|
| **Deployment** | Streamlit Community Cloud (auto-deploys on push to `main`) |
| **Snapshot cards** | Latest value + YoY delta per indicator, color-coded by economic direction |
| **Charts** | Plotly line charts with NBER recession bands (2001, 2007–09, 2020) |
| **Interactivity** | Indicator toggles, date range picker, live data refresh |
| **Data table** | Expandable raw data view with CSV download |

Key design decisions:
- Delta colors reflect whether a move is **economically good or bad**, not just whether the number went up or down (configured via `positive_direction` in `config.py`).
- YoY comparisons match on the **same calendar month** in the prior year, not a fixed N-step offset, so they're robust to data gaps and irregular spacing.
- Charts and snapshot logic are fully decoupled from the Streamlit layout in `src/chart_helpers.py`.

---

## Phase 3 Summary — DS Layer

*Completed April 2026.*

| | |
|---|---|
| **Forecasting** | Prophet 12-month forecasts per indicator, anchored to last actual value |
| **Anomaly detection** | Trailing 24-month rolling z-score (±2σ threshold) |
| **Narrative** | Rule-based plain-language findings per indicator |
| **Sidebar toggles** | Forecast, anomaly markers, and narrative each independently on/off |
| **Unit tests** | 33 new tests across forecaster, anomaly detector, and narrative modules |

Key design decisions:
- **Forecast anchoring:** Prophet's in-sample `yhat` at the last historical date rarely matches the actual observed value, especially after volatile periods. The forecast is shifted so the dashed line starts exactly where the historical line ends.
- **Quarterly series handling:** Prophet frequency is inferred from `config.py` — quarterly series use `freq="QS"` and `horizon_months // 3` periods, so no manual resampling is needed.
- **Trailing edge suppression:** The last 3 observations and any anomaly events ending within 3 months of the series end are suppressed to avoid false positives from the rolling window's lack of forward context.
- **Implausibility cap:** Forecast narratives exceeding 12% projected change for rate/percent series (25% for index series) report direction only, acknowledging that large anchor shifts from volatile recent periods make point estimates unreliable.
- **Honest long-run comparisons:** The "vs long-run average" narrative adds context caveats where the mean is structurally misleading — the Fed Funds zero-bound era skews its average down; CPI and home price indices reflect structural price growth since their base periods.
- **No LLM dependency:** The narrative layer is fully rule-based. It produces auditable, testable findings with no external API calls required at runtime.

### Known issues
- **Forecast confidence band clipping:** The forecast confidence band is partially clipped at the right edge of charts due to a Plotly 5.22 bug with `autorange` on date axes. The forecast line and direction are still clearly visible. This will be addressed when upgrading to a later Plotly version.

---

## Deploying to Streamlit Community Cloud

1. Push repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → point to `app.py`.
3. Under **Advanced settings**, set Python version to **3.12**.
4. Under **Secrets**, add:
   ```toml
   FRED_API_KEY = "your_key_here"
   ```
5. Ensure `packages.txt` is present in the repo root — this installs `libgomp1`,
   a system library required by Prophet's Stan compiler on Streamlit Cloud.
6. Deploy. The app auto-redeploys on every push to `main`.

> **Dependency note:** `requirements.txt` pins `prophet==1.1.6` and
> `cmdstanpy==1.2.4` together. These versions are a known-good combination —
> unpinning either can trigger a `'Prophet' object has no attribute 'stan_backend'`
> error due to a breaking API change in `cmdstanpy>=1.3.0`.

---

## Data Notes

- All timestamps are UTC-naive `DatetimeIndex` with index name `"date"`.
- Monthly and quarterly series are outer-joined; `NaN` in quarterly columns between observation dates is expected.
- Raw and processed CSVs are excluded from git — run the pipeline to regenerate locally.
- NBER recession dates are hardcoded constants in `src/chart_helpers.py` (they do not change retroactively).
- Prophet forecasts are computed against the full historical series regardless of the date range filter set in the sidebar, so the forecast always starts from the true last observation.
