"""
narrative.py
------------
Rule-based plain-language narrative generation for the PNW Economic Health
Dashboard's "So What?" section.

Design philosophy
-----------------
Every narrative is assembled from a set of discrete *findings* — small,
testable, self-contained observations about one indicator.  Each finding maps
to a human-readable sentence.  The final narrative is a sorted, de-duplicated
list of those sentences.

This keeps the logic auditable: if a sentence appears in the UI, you can trace
it back to a specific rule in ``_findings_for_series()``.  It also makes the
module easy to extend — add a new finding function, register it, done.

Severity vocabulary
-------------------
Delta magnitudes are bucketed as:
  - "mild":         |delta_pct| < 5 %
  - "moderate":     5 % ≤ |delta_pct| < 15 %
  - "significant":  |delta_pct| ≥ 15 %

These thresholds are calibrated for the four current indicators.  They are
internal constants; adjust ``_SEVERITY_THRESHOLDS`` if you add series with
very different variance profiles.

Usage
-----
    from src.narrative import build_narrative, build_all_narratives

    # Single series
    text = build_narrative(df, "unemployment_rate", flags, forecast_df)

    # All series → dict of key → narrative string
    narratives = build_all_narratives(df, all_flags, all_forecasts)
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from src.config import FRED_SERIES
from src.anomaly_detector import get_anomaly_events

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity thresholds for YoY delta (in percent)
# ---------------------------------------------------------------------------

_SEVERITY_THRESHOLDS: dict[str, float] = {
    "mild":     5.0,
    "moderate": 15.0,
    # anything above "moderate" → "significant"
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_narrative(
    df: pd.DataFrame,
    key: str,
    anomaly_flags: pd.Series | None = None,
    forecast_df: pd.DataFrame | None = None,
) -> list[str]:
    """
    Build a list of plain-language findings for one indicator.

    Each finding is a single sentence (no trailing period added here; the UI
    layer can style them as bullet points or paragraphs).

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame with DatetimeIndex and a column named ``key``.
    key : str
        Series key (must be in ``config.FRED_SERIES``).
    anomaly_flags : pd.Series | None
        Boolean anomaly flags from ``anomaly_detector.detect_anomalies()``.
        Pass None to skip anomaly findings.
    forecast_df : pd.DataFrame | None
        Prophet forecast DataFrame from ``forecaster.forecast_series()``.
        Pass None to skip forecast findings.

    Returns
    -------
    list[str]
        Ordered list of finding sentences.  Ordered: current status →
        historical context → anomalies → forecast direction.
        Empty list if the series has no data.
    """
    if key not in FRED_SERIES:
        logger.warning("Key '%s' not in FRED_SERIES, returning empty narrative.", key)
        return []

    meta   = FRED_SERIES[key]
    series = df[key].dropna()

    if series.empty:
        return [f"No data available for {meta['label']}."]

    findings: list[str] = []

    # --- 1. Current value & label ---
    findings += _finding_current_value(series, meta)

    # --- 2. YoY change & severity ---
    findings += _finding_yoy_change(series, meta)

    # --- 3. Position relative to long-run average ---
    findings += _finding_vs_long_run(series, meta)

    # --- 4. Anomaly events ---
    if anomaly_flags is not None:
        findings += _finding_anomalies(series, key, anomaly_flags)

    # --- 5. Forecast direction ---
    if forecast_df is not None:
        findings += _finding_forecast(series, forecast_df, meta)

    return findings


def build_all_narratives(
    df: pd.DataFrame,
    all_flags: dict[str, pd.Series] | None = None,
    all_forecasts: dict[str, pd.DataFrame] | None = None,
) -> dict[str, list[str]]:
    """
    Build narratives for every configured series in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical DataFrame.
    all_flags : dict[str, pd.Series] | None
        Anomaly flag series per key (from ``detect_all_anomalies``).
    all_forecasts : dict[str, pd.DataFrame] | None
        Forecast DataFrames per key (from ``forecast_all``).

    Returns
    -------
    dict[str, list[str]]
        Mapping of series key → list of narrative sentences.
    """
    results: dict[str, list[str]] = {}

    for key in FRED_SERIES:
        if key not in df.columns:
            continue
        flags    = (all_flags    or {}).get(key)
        forecast = (all_forecasts or {}).get(key)
        try:
            results[key] = build_narrative(df, key, flags, forecast)
        except Exception as exc:
            logger.error("Narrative generation failed for '%s': %s", key, exc)
            results[key] = [f"Could not generate narrative for this indicator."]

    return results


# ---------------------------------------------------------------------------
# Finding generators (each returns a list of 0–2 sentences)
# ---------------------------------------------------------------------------

def _finding_current_value(series: pd.Series, meta: dict) -> list[str]:
    """State the latest observed value and when it was recorded."""
    latest_val  = series.iloc[-1]
    latest_date = series.index[-1]
    label       = meta["label"]
    units       = meta["units"]
    date_str    = latest_date.strftime("%B %Y")

    return [
        f"As of {date_str}, {label} stands at {latest_val:.2f} {units}."
    ]


def _finding_yoy_change(series: pd.Series, meta: dict) -> list[str]:
    """Describe the year-over-year change in plain language with severity."""
    latest_val  = series.iloc[-1]
    latest_date = series.index[-1]
    label       = meta["label"]
    pos_dir     = meta.get("positive_direction", "up")

    # Find the value ~12 months prior (same logic as compute_snapshot)
    prior_target = latest_date - pd.DateOffset(years=1)
    window_start = prior_target - pd.Timedelta(days=45)
    window_end   = prior_target + pd.Timedelta(days=45)
    candidates   = series.loc[window_start:window_end]

    if candidates.empty:
        return ["Insufficient history to compute a year-over-year comparison."]

    time_deltas  = pd.Series(candidates.index.tolist(), index=candidates.index) - prior_target
    closest_date = time_deltas.abs().idxmin()
    prior_val    = candidates.loc[closest_date]

    if prior_val == 0:
        return []

    delta     = latest_val - prior_val
    delta_pct = abs(delta / prior_val * 100)
    severity  = _classify_severity(delta_pct)
    direction = "risen" if delta > 0 else "fallen"

    # Is this move economically good or bad?
    is_good = (delta > 0) == (pos_dir == "up")
    sentiment = "a positive sign" if is_good else "a concern worth monitoring"
    if severity == "mild":
        sentiment = "broadly stable"

    return [
        f"{label} has {direction} {delta_pct:.1f}% year-over-year — "
        f"{severity} movement and {sentiment}."
    ]


def _finding_vs_long_run(series: pd.Series, meta: dict) -> list[str]:
    """Compare latest value to the full-history mean."""
    latest_val = series.iloc[-1]
    hist_mean  = series.mean()
    label      = meta["label"]
    pos_dir    = meta.get("positive_direction", "up")

    pct_diff = (latest_val - hist_mean) / hist_mean * 100

    if abs(pct_diff) < 2.0:
        return [f"{label} is near its long-run average of {hist_mean:.2f}."]

    above_below = "above" if pct_diff > 0 else "below"
    is_good     = (pct_diff > 0) == (pos_dir == "up")
    qualifier   = "which is historically favourable" if is_good else "which is historically elevated"

    return [
        f"The current reading is {abs(pct_diff):.1f}% {above_below} the long-run average "
        f"of {hist_mean:.2f}, {qualifier}."
    ]


def _finding_anomalies(
    series: pd.Series,
    key: str,
    flags: pd.Series,
) -> list[str]:
    """Describe recent anomalous periods, if any."""
    events = get_anomaly_events(
        pd.DataFrame({key: series}),
        key,
        flags,
        n_recent=2,
    )

    if events.empty:
        return ["No statistically unusual observations detected in the recent record."]

    sentences: list[str] = []
    for _, row in events.iterrows():
        start_str = row["start"].strftime("%B %Y")
        end_str   = row["end"].strftime("%B %Y")
        direction = row["direction"]
        peak      = row["peak_value"]

        if start_str == end_str:
            period_str = start_str
        else:
            period_str = f"{start_str}–{end_str}"

        sentences.append(
            f"An unusual period was detected {period_str}: values were notably "
            f"{direction} the historical norm (peak: {peak:.2f})."
        )

    return sentences


def _finding_forecast(
    series: pd.Series,
    forecast_df: pd.DataFrame,
    meta: dict,
) -> list[str]:
    """Describe the 12-month forecast direction and confidence."""
    label   = meta["label"]
    pos_dir = meta.get("positive_direction", "up")

    last_historical = series.index[-1]

    # Isolate only the future portion of the forecast DataFrame.
    future = forecast_df[forecast_df["ds"] > last_historical].copy()

    if future.empty:
        return []

    forecast_end_val   = float(future["yhat"].iloc[-1])
    forecast_start_val = float(future["yhat"].iloc[0])
    last_actual        = float(series.iloc[-1])

    total_change     = forecast_end_val - last_actual
    total_change_pct = abs(total_change / last_actual * 100) if last_actual != 0 else 0.0

    direction = "rise" if total_change > 0 else "fall"
    is_good   = (total_change > 0) == (pos_dir == "up")
    outlook   = "a potentially positive development" if is_good else "a potential headwind"

    horizon_months = len(future)
    horizon_label  = f"{horizon_months} month{'s' if horizon_months != 1 else ''}"

    # If the projected change is implausibly large (>20% for any indicator),
    # report direction only without a specific magnitude — a large anchor shift
    # from a volatile recent period can produce unreliable point estimates.
    if total_change_pct > 20.0:
        return [
            f"The model projects {label} to {direction} over the next {horizon_label}, "
            f"though the magnitude is uncertain given recent volatility."
        ]

    return [
        f"The model projects {label} to {direction} approximately {total_change_pct:.1f}% "
        f"over the next {horizon_label}, {outlook}."
    ]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _classify_severity(delta_pct: float) -> str:
    """Bucket an absolute percentage change into a severity label."""
    if delta_pct < _SEVERITY_THRESHOLDS["mild"]:
        return "mild"
    elif delta_pct < _SEVERITY_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "significant"
