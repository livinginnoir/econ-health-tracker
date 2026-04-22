"""
config.py
---------
Central configuration for the PNW Regional Economic Health Dashboard.

All FRED series IDs, human-readable labels, and shared settings live here.
Import this module in any other module that needs series metadata — do not
hardcode IDs or labels elsewhere.
"""

# ---------------------------------------------------------------------------
# FRED Series Definitions
# ---------------------------------------------------------------------------
# Each entry maps a short internal key to its FRED series ID and display name.
# Add new series here; the rest of the pipeline picks them up automatically.

FRED_SERIES: dict[str, dict] = {
    "unemployment_rate": {
        "series_id": "ORUR",
        "label": "Oregon Unemployment Rate",
        "units": "Percent",
        "frequency": "Monthly",
        "source": "FRED",
        "positive_direction": "down",   # lower unemployment = better
        "value_floor": 0.0,             # Prophet yhat_lower clamp (Phase 3)
    },
    "cpi_west": {
        "series_id": "CUURA400SA0",
        "label": "CPI – Western Urban Areas (All Items)",
        "units": "Index (1982–84=100)",
        "frequency": "Monthly",
        "source": "FRED",
        "positive_direction": "down",   # lower inflation = better
        "value_floor": 0.0,
    },
    "home_price_index": {
        "series_id": "ATNHPIUS38900Q",
        "label": "Portland Metro Home Price Index",
        "units": "Index",
        "frequency": "Quarterly",
        "source": "FRED",
        "positive_direction": "down",   # lower = more affordable
        "value_floor": 0.0,
    },
    "fed_funds_rate": {
        "series_id": "FEDFUNDS",
        "label": "Federal Funds Rate",
        "units": "Percent",
        "frequency": "Monthly",
        "source": "FRED",
        "positive_direction": "down",   # lower rates = looser conditions
        "value_floor": 0.0,
    },
}

# ---------------------------------------------------------------------------
# BLS Series Definitions (Phase 1 placeholder — expand as needed)
# ---------------------------------------------------------------------------
# BLS series IDs follow the BLS public data API v2 format.
# See: https://www.bls.gov/developers/api_faqs.htm

BLS_SERIES: dict[str, dict] = {
    # Example: Portland-area total nonfarm employment (CES series)
    # Uncomment and add your target series IDs here.
    # "portland_nonfarm_employment": {
    #     "series_id": "SMU41389000000000001",
    #     "label": "Portland Metro Total Nonfarm Employment",
    #     "units": "Thousands of Persons",
    #     "frequency": "Monthly",
    #     "source": "BLS",
    # },
}

# ---------------------------------------------------------------------------
# Data Fetching Settings
# ---------------------------------------------------------------------------

# Default date range for all FRED pulls (YYYY-MM-DD strings).
# Set DEFAULT_START_DATE far enough back for meaningful trend analysis.
DEFAULT_START_DATE: str = "2000-01-01"
DEFAULT_END_DATE: str | None = None  # None → FRED returns up to today

# Local cache directory for raw CSVs (excluded from git via .gitignore).
RAW_DATA_DIR: str = "data/raw"
PROCESSED_DATA_DIR: str = "data/processed"

# ---------------------------------------------------------------------------
# Phase 3 — Forecasting Settings
# ---------------------------------------------------------------------------

# Number of months to forecast ahead.  Prophet converts this to the correct
# number of periods for quarterly series automatically via freq inference.
FORECAST_HORIZON_MONTHS: int = 12

# ---------------------------------------------------------------------------
# Phase 3 — Anomaly Detection Settings
# ---------------------------------------------------------------------------

# Rolling window size (in observations) for the z-score baseline.
# 24 months = 2 full years; long enough to capture a business cycle segment
# without being so long that the window becomes insensitive to regime shifts.
ANOMALY_ZSCORE_WINDOW: int = 24

# Number of standard deviations beyond which an observation is flagged.
# 2.0 σ ≈ top/bottom 2.3% of a normal distribution — a reasonable signal
# threshold that avoids excessive false positives on economic series.
ANOMALY_ZSCORE_THRESH: float = 2.0

# ---------------------------------------------------------------------------
# Streamlit App Settings
# ---------------------------------------------------------------------------

APP_TITLE: str = "PNW Regional Economic Health Dashboard"
APP_SUBTITLE: str = "Portland & Pacific Northwest Key Indicators"
