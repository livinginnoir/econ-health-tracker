# PNW Regional Economic Health Dashboard

An interactive dashboard tracking key economic indicators for Portland and the Pacific Northwest. Built as a portfolio project demonstrating data engineering, data science, and business communication skills.

**Live app:** *(link after Streamlit Community Cloud deploy)*  
**Tech stack:** Python · FRED API · BLS API · Pandas · Prophet · Streamlit

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pipeline (FRED + BLS ingestion, CSV caching) | ✅ Complete |
| 2 | Core Streamlit dashboard (visualisations, interactivity) | 🔜 Up next |
| 3 | DS layer (Prophet forecasting, anomaly detection, "So What?" narrative) | 🔜 Planned |

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

### 4. Run tests
```bash
pytest tests/ -v
```

---

## Project Structure

```
pnw_dashboard/
├── src/
│   ├── config.py          # Series IDs, labels, settings
│   └── data_fetcher.py    # FRED + BLS ingestion, caching
├── data/
│   ├── raw/               # Cached individual series CSVs (gitignored)
│   └── processed/         # Merged output for dashboard
├── tests/
│   └── test_data_fetcher.py
├── run_pipeline.py        # Phase 1 pipeline entry point
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Data Notes

- Series with different frequencies (monthly vs quarterly) are outer-joined — sparse values in quarterly columns are expected and handled in the dashboard layer.
- All timestamps are UTC-naive `DatetimeIndex` with index name `"date"`.
- Raw CSVs are excluded from git; run the pipeline to regenerate locally.
