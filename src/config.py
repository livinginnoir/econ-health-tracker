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
    },
    "cpi_west": {
        "series_id": "CUURA400SA0",
        "label": "CPI – Western Urban Areas (All Items)",
        "units": "Index (1982–84=100)",
        "frequency": "Monthly",
        "source": "FRED",
    },
    "home_price_index": {
        "series_id": "ATNHPIUS38900Q",
        "label": "Portland Metro Home Price Index",
        "units": "Index",
        "frequency": "Quarterly",
        "source": "FRED",
    },
    "fed_funds_rate": {
        "series_id": "FEDFUNDS",
        "label": "Federal Funds Rate",
        "units": "Percent",
        "frequency": "Monthly",
        "source": "FRED",
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
# Streamlit App Settings (used in Phase 2+)
# ---------------------------------------------------------------------------

APP_TITLE: str = "PNW Regional Economic Health Dashboard"
APP_SUBTITLE: str = "Portland & Pacific Northwest Key Indicators"
