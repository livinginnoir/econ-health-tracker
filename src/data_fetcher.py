"""
data_fetcher.py
---------------
Reusable data ingestion module for the PNW Regional Economic Health Dashboard.

Handles:
  - FRED API pulls via `fredapi`
  - BLS API pulls via direct HTTP requests (public API v2)
  - Local CSV caching to avoid redundant API calls during development
  - Light normalization so all returned DataFrames share a consistent schema

Schema contract for all returned DataFrames:
  - Index: DatetimeIndex, UTC-naive, named "date"
  - One column per series, named by the internal key defined in config.py
  - Column dtype: float64
  - No duplicate index entries

Usage
-----
    from src.data_fetcher import fetch_fred_series, fetch_all_fred, fetch_bls_series

    # Fetch a single FRED series
    df = fetch_fred_series("unemployment_rate")

    # Fetch all configured FRED series into one merged DataFrame
    df_all = fetch_all_fred()
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

from src.config import (
    BLS_SERIES,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    FRED_SERIES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Module-level logger — callers can configure the root logger as needed.
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Environment & API client initialisation
# ---------------------------------------------------------------------------
# Load .env from the project root (one level up from src/).
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

_FRED_API_KEY = os.getenv("FRED_API_KEY")
_BLS_API_KEY = os.getenv("BLS_API_KEY")  # Optional; BLS v2 works without a key (rate-limited)

if not _FRED_API_KEY:
    raise EnvironmentError(
        "FRED_API_KEY not found. "
        "Copy .env.example → .env and add your key from https://fred.stlouisfed.org/docs/api/api_key.html"
    )

# Instantiate the FRED client once at module level (thread-safe for reads).
_fred = Fred(api_key=_FRED_API_KEY)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str, subdir: str = RAW_DATA_DIR) -> Path:
    """Return the local CSV path for a given series key."""
    return Path(subdir) / f"{key}.csv"


def _save_to_cache(df: pd.DataFrame, key: str, subdir: str = RAW_DATA_DIR) -> None:
    """Persist a DataFrame to CSV cache. Creates directories as needed."""
    path = _cache_path(key, subdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.debug("Cached %s → %s", key, path)


def _load_from_cache(key: str, subdir: str = RAW_DATA_DIR) -> pd.DataFrame | None:
    """
    Load a DataFrame from CSV cache if it exists.

    Returns None if no cache file is found, so callers can fall back to the API.
    """
    path = _cache_path(key, subdir)
    if path.exists():
        logger.info("Loading '%s' from cache (%s)", key, path)
        df = pd.read_csv(path, index_col="date", parse_dates=True)
        return df
    return None


# ---------------------------------------------------------------------------
# FRED fetching
# ---------------------------------------------------------------------------

def fetch_fred_series(
    key: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch a single FRED series by its internal config key.

    Parameters
    ----------
    key : str
        Internal key from config.FRED_SERIES (e.g. "unemployment_rate").
    start_date : str
        ISO date string for the start of the pull window (default: config value).
    end_date : str | None
        ISO date string for the end of the pull window. None → today.
    use_cache : bool
        If True, return cached CSV if available (skips API call).
        Useful during development; set False for production freshness.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with DatetimeIndex named "date" and column = key.

    Raises
    ------
    KeyError
        If `key` is not found in config.FRED_SERIES.
    """
    if key not in FRED_SERIES:
        raise KeyError(
            f"'{key}' not found in FRED_SERIES config. "
            f"Available keys: {list(FRED_SERIES.keys())}"
        )

    meta = FRED_SERIES[key]
    series_id = meta["series_id"]

    # --- Cache check ---
    if use_cache:
        cached = _load_from_cache(key)
        if cached is not None:
            return cached

    logger.info(
        "Fetching FRED series '%s' (%s) from %s to %s",
        series_id,
        meta["label"],
        start_date,
        end_date or "today",
    )

    # --- API pull ---
    raw: pd.Series = _fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date,
    )

    # --- Normalise to standard DataFrame schema ---
    df = _normalise_fred_series(raw, key)

    # --- Cache to disk ---
    _save_to_cache(df, key)

    return df


def _normalise_fred_series(raw: pd.Series, key: str) -> pd.DataFrame:
    """
    Convert a raw fredapi Series into the dashboard's standard DataFrame schema.

    - Drops NaN rows
    - Ensures DatetimeIndex is UTC-naive and named "date"
    - Names the value column after the internal key
    - Casts values to float64
    """
    df = raw.to_frame(name=key)
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip any tz info
    df.index.name = "date"
    df = df.dropna()
    df[key] = df[key].astype("float64")
    df = df[~df.index.duplicated(keep="last")]  # deduplicate (rare but safe)
    df.sort_index(inplace=True)
    return df


def fetch_all_fred(
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch every FRED series defined in config.FRED_SERIES and merge into one DataFrame.

    Series with different frequencies (e.g. monthly vs quarterly) are outer-joined
    on their date index — sparse columns are expected and handled downstream.

    Returns
    -------
    pd.DataFrame
        Multi-column DataFrame with DatetimeIndex named "date".
        Columns correspond to config.FRED_SERIES keys.
    """
    frames: list[pd.DataFrame] = []

    for key in FRED_SERIES:
        try:
            df = fetch_fred_series(key, start_date, end_date, use_cache)
            frames.append(df)
        except Exception as exc:
            # Log and continue — a single series failure shouldn't kill the pipeline.
            logger.error("Failed to fetch FRED series '%s': %s", key, exc)

    if not frames:
        raise RuntimeError("No FRED series could be fetched. Check your API key and network.")

    # Outer join preserves all timestamps across frequencies.
    merged = pd.concat(frames, axis=1, join="outer")
    merged.sort_index(inplace=True)
    logger.info(
        "Merged %d FRED series into DataFrame with shape %s", len(frames), merged.shape
    )
    return merged


# ---------------------------------------------------------------------------
# BLS fetching
# ---------------------------------------------------------------------------

_BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"


def fetch_bls_series(
    key: str,
    start_year: int = 2000,
    end_year: int | None = None,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch a single BLS series by its internal config key.

    Uses the BLS Public Data API v2. An API key is optional but increases rate limits
    from 25 to 500 queries/day — set BLS_API_KEY in .env to use it.

    Parameters
    ----------
    key : str
        Internal key from config.BLS_SERIES.
    start_year : int
        First year to pull. BLS API allows max 20-year spans per request.
    end_year : int | None
        Last year to pull. None → current year.
    use_cache : bool
        Return cached CSV if available.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with DatetimeIndex named "date" and column = key.

    Raises
    ------
    KeyError
        If `key` is not in config.BLS_SERIES.
    RuntimeError
        If the BLS API returns an error status.
    """
    if key not in BLS_SERIES:
        raise KeyError(
            f"'{key}' not found in BLS_SERIES config. "
            f"Available keys: {list(BLS_SERIES.keys())}"
        )

    if use_cache:
        cached = _load_from_cache(key)
        if cached is not None:
            return cached

    import datetime
    if end_year is None:
        end_year = datetime.date.today().year

    meta = BLS_SERIES[key]
    series_id = meta["series_id"]

    logger.info(
        "Fetching BLS series '%s' (%s) %d–%d",
        series_id,
        meta["label"],
        start_year,
        end_year,
    )

    payload: dict = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "calculations": False,
        "annualaverage": False,
    }
    if _BLS_API_KEY:
        payload["registrationkey"] = _BLS_API_KEY

    response = requests.post(_BLS_API_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()

    if data.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(
            f"BLS API error for series '{series_id}': {data.get('message', data)}"
        )

    series_data = data["Results"]["series"][0]["data"]
    df = _normalise_bls_series(series_data, key)
    _save_to_cache(df, key)
    return df


def _normalise_bls_series(records: list[dict], key: str) -> pd.DataFrame:
    """
    Convert raw BLS API records into the dashboard's standard DataFrame schema.

    BLS returns a list of dicts with keys: year, period, periodName, value.
    Monthly periods look like "M01"–"M12"; annual is "M13" (skipped).
    """
    rows = []
    for rec in records:
        period = rec.get("period", "")
        if not period.startswith("M") or period == "M13":
            continue  # skip annual averages and non-monthly records
        month = int(period[1:])
        year = int(rec["year"])
        value = float(rec["value"].replace(",", ""))
        rows.append({"date": pd.Timestamp(year=year, month=month, day=1), key: value})

    if not rows:
        raise ValueError("No monthly records found in BLS response.")

    df = pd.DataFrame(rows).set_index("date")
    df.index.name = "date"
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df


# ---------------------------------------------------------------------------
# Processed data helpers (Phase 2+ convenience)
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, filename: str) -> Path:
    """
    Save a processed DataFrame to the processed data directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save (must have DatetimeIndex named "date").
    filename : str
        Output filename, e.g. "all_indicators.csv".

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    out_dir = Path(PROCESSED_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    df.to_csv(out_path)
    logger.info("Saved processed data → %s", out_path)
    return out_path


def load_processed(filename: str) -> pd.DataFrame:
    """
    Load a processed CSV from the processed data directory.

    Parameters
    ----------
    filename : str
        Filename within the processed data directory.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex named "date".

    Raises
    ------
    FileNotFoundError
        If the processed file does not exist (run the pipeline first).
    """
    path = Path(PROCESSED_DATA_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            "Run the pipeline script to generate it."
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)
