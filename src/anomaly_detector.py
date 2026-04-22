"""
anomaly_detector.py
-------------------
Rolling z-score anomaly detection for the PNW Regional Economic Health Dashboard.

Approach
--------
For each observation, we compute the z-score relative to a *trailing* rolling
window of the previous N observations.  An observation is flagged as anomalous
if its z-score exceeds ±ANOMALY_ZSCORE_THRESH.

Why trailing (not centred)?
  A centred window "looks ahead" — it uses future data to evaluate the past.
  That works fine for static historical analysis but breaks down conceptually
  when you're building a dashboard meant to flag real-time conditions.  A
  trailing window is honest: it only compares each point to what was known
  before it.

Why rolling z-score instead of IQR or DBSCAN?
  Economic series are non-stationary (trending up over decades).  A global
  z-score would flag the entire post-2000 CPI run as anomalous.  A rolling
  window adapts to the local level and scale, which surfaces event-driven
  shocks (COVID spike, GFC) rather than structural trends.

Schema contract
---------------
All public functions accept a ``pd.DataFrame`` with DatetimeIndex named "date"
and return a boolean ``pd.Series`` (or dict of them) with the same index.
``True`` means "flagged as anomalous".  NaN positions are always ``False``.

Usage
-----
    from src.anomaly_detector import detect_anomalies, detect_all_anomalies

    flags = detect_anomalies(df, "unemployment_rate")
    all_flags = detect_all_anomalies(df)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import (
    ANOMALY_ZSCORE_THRESH,
    ANOMALY_ZSCORE_WINDOW,
    FRED_SERIES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    key: str,
    window: int = ANOMALY_ZSCORE_WINDOW,
    threshold: float = ANOMALY_ZSCORE_THRESH,
) -> pd.Series:
    """
    Flag anomalous observations in one indicator series using a rolling z-score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex and a column named ``key``.
    key : str
        Column name to analyse.
    window : int
        Number of trailing observations for the rolling baseline
        (default: config.ANOMALY_ZSCORE_WINDOW = 24).
    threshold : float
        Z-score magnitude above which a point is flagged
        (default: config.ANOMALY_ZSCORE_THRESH = 2.0).

    Returns
    -------
    pd.Series
        Boolean Series with the same index as ``df[key].dropna()``.
        True = flagged as anomalous.  Named ``f"{key}_anomaly"``.
    """
    if key not in df.columns:
        raise KeyError(f"Column '{key}' not found in DataFrame.")

    series = df[key].dropna()

    if len(series) < window:
        logger.warning(
            "Series '%s' has %d observations, fewer than the window size %d. "
            "Z-scores will be computed on whatever data is available.",
            key, len(series), window,
        )

    # Rolling mean and std over the *trailing* window.
    # min_periods ensures we still get values early in the series (at the cost
    # of a wider effective window), but we require at least 3 observations to
    # avoid division by near-zero std at startup.
    rolling_mean = series.rolling(window=window, min_periods=3).mean()
    rolling_std  = series.rolling(window=window, min_periods=3).std()

    # Avoid division by zero for flat series (e.g. fed funds rate at 0 for years).
    # A std of zero means no variation → no anomaly possible.
    safe_std = rolling_std.replace(0, float("nan"))

    z_scores = (series - rolling_mean) / safe_std

    flags = z_scores.abs() > threshold

    # NaN z-scores (first few observations, or flat windows) → not flagged.
    flags = flags.fillna(False)
    flags.name = f"{key}_anomaly"

    n_flagged = int(flags.sum())
    logger.info(
        "Anomaly detection for '%s': %d/%d observations flagged (threshold=±%.1fσ).",
        key, n_flagged, len(series), threshold,
    )

    return flags


def detect_all_anomalies(
    df: pd.DataFrame,
    window: int = ANOMALY_ZSCORE_WINDOW,
    threshold: float = ANOMALY_ZSCORE_THRESH,
) -> dict[str, pd.Series]:
    """
    Run anomaly detection on every configured FRED series present in ``df``.

    Series that fail are logged and skipped; the return dict will simply not
    contain that key.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame from ``data_fetcher.fetch_all_fred()``.
    window : int
        Trailing window size (passed through to ``detect_anomalies``).
    threshold : float
        Z-score threshold (passed through to ``detect_anomalies``).

    Returns
    -------
    dict[str, pd.Series]
        Mapping of series key → boolean anomaly flags.
    """
    results: dict[str, pd.Series] = {}

    for key in FRED_SERIES:
        if key not in df.columns:
            logger.warning("Key '%s' not in DataFrame, skipping anomaly detection.", key)
            continue
        try:
            results[key] = detect_anomalies(df, key, window, threshold)
        except Exception as exc:
            logger.error("Anomaly detection failed for '%s': %s", key, exc)

    return results


# ---------------------------------------------------------------------------
# Convenience: annotated anomaly table (useful for "So What?" narrative)
# ---------------------------------------------------------------------------

def get_anomaly_events(
    df: pd.DataFrame,
    key: str,
    flags: pd.Series,
    n_recent: int = 5,
) -> pd.DataFrame:
    """
    Return a tidy table of flagged anomaly events for one series.

    Consecutive flagged observations are collapsed into a single "event" with
    a start date, end date, peak absolute value, and direction (above/below
    the rolling mean at that time).  This is the raw material the narrative
    module uses to describe anomalous periods in plain language.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame.
    key : str
        Series key.
    flags : pd.Series
        Boolean anomaly flags from ``detect_anomalies``.
    n_recent : int
        Maximum number of most-recent events to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``start``, ``end``, ``peak_value``, ``direction``,
        ``duration_months``.  Empty DataFrame if no anomalies.
    """
    series = df[key].dropna()
    flagged_dates = flags[flags].index.tolist()

    if not flagged_dates:
        return pd.DataFrame(
            columns=["start", "end", "peak_value", "direction", "duration_months"]
        )

    # --- Group consecutive flagged dates into events ---
    # Two dates are "consecutive" if they are adjacent observations in the
    # (possibly irregular) series index.  We compare positional neighbours
    # rather than calendar distance so quarterly and monthly series behave
    # the same way.
    all_dates  = series.index.tolist()
    date_pos   = {d: i for i, d in enumerate(all_dates)}

    events: list[dict] = []
    group: list[pd.Timestamp] = [flagged_dates[0]]

    for date in flagged_dates[1:]:
        prev = group[-1]
        # Consecutive if they are adjacent in the full series index.
        if date_pos.get(date, -1) == date_pos.get(prev, -2) + 1:
            group.append(date)
        else:
            events.append(_summarise_event(group, series))
            group = [date]
    events.append(_summarise_event(group, series))

    events_df = pd.DataFrame(events)
    # Return the N most recent events.
    return events_df.sort_values("start", ascending=False).head(n_recent).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _summarise_event(
    group: list[pd.Timestamp],
    series: pd.Series,
) -> dict:
    """Collapse a list of consecutive anomalous dates into a summary dict."""
    values         = series.loc[group]
    peak_idx       = values.abs().idxmax()
    peak_value     = float(values.loc[peak_idx])
    series_mean    = float(series.mean())
    direction      = "above" if peak_value > series_mean else "below"
    duration_months = len(group)  # approximate; exact for monthly, ~3× for quarterly

    return {
        "start":           group[0],
        "end":             group[-1],
        "peak_value":      peak_value,
        "direction":       direction,
        "duration_months": duration_months,
    }
