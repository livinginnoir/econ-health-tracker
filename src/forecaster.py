"""
forecaster.py
-------------
Prophet-based forecasting for the PNW Regional Economic Health Dashboard.

Each public function accepts a tidy ``pd.DataFrame`` (DatetimeIndex, one column
per key) and returns a forecast DataFrame in a schema the chart helpers can
consume directly.

Design notes
------------
- Prophet requires a DataFrame with columns ``ds`` (datetime) and ``y`` (float).
  We convert to/from that schema internally; callers never touch it.
- Quarterly series (``frequency == "Quarterly"`` in config) are handled by
  passing ``freq="QS"`` to ``make_future_dataframe``.  Prophet infers the
  period from the training data, but we make it explicit to avoid edge-case
  misdetection on sparse series.
- ``yhat_lower`` is clamped to ``value_floor`` (default 0.0) so negative
  unemployment or price index values never appear in the UI.
- Suppresses Prophet's verbose Stan output via Python's logging module — users
  should see clean Streamlit output.
- All public functions are pure (no side effects) so they compose cleanly
  with ``@st.cache_data``.
"""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet

from src.config import FORECAST_HORIZON_MONTHS, FRED_SERIES

# Suppress Prophet/Stan's extremely chatty log output.
# Set to logging.INFO temporarily if you need to debug a model fit.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_series(
    df: pd.DataFrame,
    key: str,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
) -> pd.DataFrame:
    """
    Fit a Prophet model on one indicator series and return a forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame with DatetimeIndex named "date" and a column
        named ``key``.  Only rows where ``key`` is non-null are used for fitting.
    key : str
        Column name in ``df`` — must exist in ``config.FRED_SERIES``.
    horizon_months : int
        Number of *calendar months* to forecast ahead (default: config value).
        For quarterly series, this is converted to the equivalent number of
        quarterly periods automatically.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds`` (datetime), ``yhat``, ``yhat_lower``, ``yhat_upper``.
        The DataFrame spans from the first historical observation through the
        end of the forecast horizon, so it can be overlaid on the full chart.

    Raises
    ------
    KeyError
        If ``key`` is not found in ``config.FRED_SERIES``.
    ValueError
        If the series has fewer than 24 non-null observations (too short to
        fit a meaningful Prophet model).
    """
    if key not in FRED_SERIES:
        raise KeyError(
            f"'{key}' not found in FRED_SERIES. "
            f"Available keys: {list(FRED_SERIES.keys())}"
        )

    meta = FRED_SERIES[key]
    series = df[key].dropna()

    if len(series) < 24:
        raise ValueError(
            f"Series '{key}' has only {len(series)} observations. "
            "At least 24 are required to fit a Prophet model."
        )

    # --- Determine forecast frequency ---
    # Prophet's make_future_dataframe needs a period count, not a month count.
    is_quarterly = meta.get("frequency", "Monthly").lower() == "quarterly"
    freq         = "QS" if is_quarterly else "MS"   # quarter-start / month-start
    periods      = horizon_months // 3 if is_quarterly else horizon_months

    # --- Build Prophet input DataFrame ---
    prophet_df = _to_prophet_df(series)

    # --- Fit model ---
    # yearly_seasonality=True captures annual cycles (e.g. seasonal unemployment).
    # weekly_seasonality and daily_seasonality are irrelevant at monthly/quarterly
    # frequency and are disabled to keep the model clean.
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        # Wider uncertainty intervals (default 0.80) give a more honest picture
        # for economic series where forecast variance compounds quickly.
        interval_width=0.90,
    )

    logger.info("Fitting Prophet model for '%s' (%d observations)…", key, len(prophet_df))
    model.fit(prophet_df)

    # --- Build future DataFrame and predict ---
    future    = model.make_future_dataframe(periods=periods, freq=freq)
    forecast  = model.predict(future)

    # --- Post-process ---
    floor = meta.get("value_floor", 0.0)
    result = _from_prophet_forecast(forecast, floor=floor)

    logger.info(
        "Forecast for '%s' complete: %d historical + %d forecast rows.",
        key,
        len(series),
        periods,
    )
    return result


def forecast_all(
    df: pd.DataFrame,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
) -> dict[str, pd.DataFrame]:
    """
    Fit and return Prophet forecasts for every series in ``df``.

    Series that fail (too short, model error) are logged and skipped — the
    return dict will simply not contain that key.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame from ``data_fetcher.fetch_all_fred()``.
    horizon_months : int
        Forecast horizon in calendar months.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of series key → forecast DataFrame (same schema as
        ``forecast_series`` return value).
    """
    results: dict[str, pd.DataFrame] = {}

    for key in FRED_SERIES:
        if key not in df.columns:
            logger.warning("Key '%s' not found in DataFrame columns, skipping.", key)
            continue
        try:
            results[key] = forecast_series(df, key, horizon_months)
        except Exception as exc:
            logger.error("Forecast failed for '%s': %s", key, exc)

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
    """
    Convert a named pandas Series with DatetimeIndex into Prophet's ``ds``/``y``
    format.  The index is cast to timezone-naive UTC to avoid Prophet warnings.
    """
    prophet_df = series.reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    return prophet_df


def _from_prophet_forecast(forecast: pd.DataFrame, floor: float = 0.0) -> pd.DataFrame:
    """
    Extract and clean the columns we need from Prophet's raw output.

    - Selects only ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    - Clamps ``yhat_lower`` to ``floor`` so the band never goes negative for
      series like rates or price indices.
    - Resets index so the result is a plain integer-indexed DataFrame (easier
      to join / filter in chart helpers).
    """
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["yhat_lower"] = result["yhat_lower"].clip(lower=floor)
    result["ds"] = pd.to_datetime(result["ds"])
    result = result.reset_index(drop=True)
    return result
