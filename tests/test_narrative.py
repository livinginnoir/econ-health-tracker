"""
tests/test_narrative.py
-----------------------
Unit tests for src/narrative.py.

All tests use synthetic DataFrames — no API key or network access required.

Run with:
    pytest tests/test_narrative.py -v
"""

import pytest
import pandas as pd
import numpy as np

from src.narrative import (
    build_narrative,
    build_all_narratives,
    _classify_severity,
    _finding_current_value,
    _finding_yoy_change,
    _finding_vs_long_run,
    _finding_forecast,
)
from src.config import FRED_SERIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_monthly_df(
    key: str = "unemployment_rate",
    n: int = 60,
    start: str = "2019-01-01",
    trend: float = 0.0,
) -> pd.DataFrame:
    """Synthetic monthly DataFrame for a single key."""
    rng    = np.random.default_rng(42)
    idx    = pd.date_range(start=start, periods=n, freq="MS")
    values = 5.0 + np.linspace(0, trend, n) + rng.normal(0, 0.2, n)
    return pd.DataFrame({key: values}, index=idx)


def make_full_df(n: int = 60) -> pd.DataFrame:
    """All four configured keys."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "unemployment_rate": 5.0 + rng.normal(0, 0.3, n),
            "cpi_west":          260.0 + rng.normal(0, 1.0, n),
            "fed_funds_rate":    2.5 + rng.normal(0, 0.1, n),
            "home_price_index":  180.0 + rng.normal(0, 2.0, n),
        },
        index=idx,
    )


def make_mock_forecast(key: str, df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Minimal mock forecast DataFrame (flat yhat = last actual)."""
    last_date  = df[key].dropna().index[-1]
    last_val   = float(df[key].dropna().iloc[-1])
    future_idx = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )
    # Combine one overlap point + future
    all_dates = [last_date] + list(future_idx)
    return pd.DataFrame({
        "ds":          all_dates,
        "yhat":        [last_val] * len(all_dates),
        "yhat_lower":  [last_val - 0.5] * len(all_dates),
        "yhat_upper":  [last_val + 0.5] * len(all_dates),
    })


def make_boolean_flags(df: pd.DataFrame, key: str, flag_pct: float = 0.0) -> pd.Series:
    """Return a boolean flag Series with ``flag_pct``% of observations flagged."""
    series    = df[key].dropna()
    flags     = pd.Series(False, index=series.index, name=f"{key}_anomaly")
    n_to_flag = int(len(flags) * flag_pct)
    if n_to_flag > 0:
        flags.iloc[:n_to_flag] = True
    return flags


# ---------------------------------------------------------------------------
# _classify_severity
# ---------------------------------------------------------------------------

class TestClassifySeverity:
    def test_mild(self):
        assert _classify_severity(2.0) == "mild"

    def test_boundary_mild_to_moderate(self):
        assert _classify_severity(5.0) == "moderate"

    def test_moderate(self):
        assert _classify_severity(10.0) == "moderate"

    def test_boundary_moderate_to_significant(self):
        assert _classify_severity(15.0) == "significant"

    def test_significant(self):
        assert _classify_severity(50.0) == "significant"

    def test_zero(self):
        assert _classify_severity(0.0) == "mild"


# ---------------------------------------------------------------------------
# _finding_current_value
# ---------------------------------------------------------------------------

class TestFindingCurrentValue:
    def test_returns_list_of_strings(self):
        df     = make_monthly_df()
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_current_value(df["unemployment_rate"].dropna(), meta)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_contains_label(self):
        df     = make_monthly_df()
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_current_value(df["unemployment_rate"].dropna(), meta)
        assert meta["label"] in result[0]

    def test_contains_value(self):
        df     = make_monthly_df()
        series = df["unemployment_rate"].dropna()
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_current_value(series, meta)
        # The latest value should appear formatted in the sentence
        assert f"{series.iloc[-1]:.2f}" in result[0]


# ---------------------------------------------------------------------------
# _finding_yoy_change
# ---------------------------------------------------------------------------

class TestFindingYoyChange:
    def test_returns_list(self):
        df     = make_monthly_df()
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_yoy_change(df["unemployment_rate"].dropna(), meta)
        assert isinstance(result, list)

    def test_insufficient_history(self):
        """Series shorter than 12 months should mention insufficient history."""
        idx    = pd.date_range("2024-01-01", periods=6, freq="MS")
        series = pd.Series([5.0] * 6, index=idx)
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_yoy_change(series, meta)
        assert len(result) == 1
        assert "insufficient" in result[0].lower()

    def test_rising_series_says_risen(self):
        df     = make_monthly_df(trend=3.0)  # strong upward trend
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_yoy_change(df["unemployment_rate"].dropna(), meta)
        assert any("risen" in s for s in result)

    def test_falling_series_says_fallen(self):
        df     = make_monthly_df(trend=-3.0)
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_yoy_change(df["unemployment_rate"].dropna(), meta)
        assert any("fallen" in s for s in result)


# ---------------------------------------------------------------------------
# _finding_vs_long_run
# ---------------------------------------------------------------------------

class TestFindingVsLongRun:
    def test_returns_list(self):
        df     = make_monthly_df()
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_vs_long_run(df["unemployment_rate"].dropna(), meta)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_near_average_wording(self):
        """A flat series should say 'near its long-run average'."""
        idx    = pd.date_range("2019-01-01", periods=60, freq="MS")
        series = pd.Series([5.0] * 60, index=idx)
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_vs_long_run(series, meta)
        assert "near its long-run average" in result[0]

    def test_above_wording(self):
        """Series that ends well above the mean should say 'above'."""
        idx    = pd.date_range("2019-01-01", periods=60, freq="MS")
        values = [5.0] * 55 + [50.0] * 5  # last 5 obs dramatically high
        series = pd.Series(values, index=idx)
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_vs_long_run(series, meta)
        assert "above" in result[0]


# ---------------------------------------------------------------------------
# _finding_forecast
# ---------------------------------------------------------------------------

class TestFindingForecast:
    def test_returns_list(self):
        df          = make_monthly_df()
        forecast_df = make_mock_forecast("unemployment_rate", df)
        meta        = FRED_SERIES["unemployment_rate"]
        result      = _finding_forecast(df["unemployment_rate"].dropna(), forecast_df, meta)
        assert isinstance(result, list)

    def test_no_future_rows_returns_empty(self):
        """If forecast_df has no rows after last historical date, return []."""
        df          = make_monthly_df()
        series      = df["unemployment_rate"].dropna()
        last_date   = series.index[-1]
        forecast_df = pd.DataFrame({
            "ds":         [last_date - pd.DateOffset(months=1)],
            "yhat":       [5.0],
            "yhat_lower": [4.5],
            "yhat_upper": [5.5],
        })
        meta   = FRED_SERIES["unemployment_rate"]
        result = _finding_forecast(series, forecast_df, meta)
        assert result == []

    def test_contains_rise_or_fall(self):
        df          = make_monthly_df()
        forecast_df = make_mock_forecast("unemployment_rate", df)
        meta        = FRED_SERIES["unemployment_rate"]
        result      = _finding_forecast(df["unemployment_rate"].dropna(), forecast_df, meta)
        text        = " ".join(result)
        assert "rise" in text or "fall" in text


# ---------------------------------------------------------------------------
# build_narrative — integration
# ---------------------------------------------------------------------------

class TestBuildNarrative:
    def test_returns_list_of_strings(self):
        df     = make_monthly_df()
        result = build_narrative(df, "unemployment_rate")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_non_empty_for_valid_series(self):
        df     = make_monthly_df()
        result = build_narrative(df, "unemployment_rate")
        assert len(result) > 0

    def test_invalid_key_returns_empty(self):
        df     = make_monthly_df()
        result = build_narrative(df, "nonexistent_key")
        assert result == []

    def test_empty_series_returns_no_data_message(self):
        idx    = pd.date_range("2023-01-01", periods=5, freq="MS")
        df     = pd.DataFrame({"unemployment_rate": [np.nan] * 5}, index=idx)
        result = build_narrative(df, "unemployment_rate")
        assert len(result) == 1
        assert "No data" in result[0]

    def test_with_forecast_adds_finding(self):
        df          = make_monthly_df()
        forecast_df = make_mock_forecast("unemployment_rate", df)
        without     = build_narrative(df, "unemployment_rate")
        with_fc     = build_narrative(df, "unemployment_rate", forecast_df=forecast_df)
        assert len(with_fc) >= len(without)

    def test_with_anomaly_flags_adds_finding(self):
        df    = make_monthly_df()
        flags = make_boolean_flags(df, "unemployment_rate", flag_pct=0.1)
        result = build_narrative(df, "unemployment_rate", anomaly_flags=flags)
        # At least one sentence should reference the anomaly period
        assert any("unusual" in s.lower() or "detected" in s.lower() for s in result)


# ---------------------------------------------------------------------------
# build_all_narratives
# ---------------------------------------------------------------------------

class TestBuildAllNarratives:
    def test_returns_dict(self):
        df     = make_full_df()
        result = build_all_narratives(df)
        assert isinstance(result, dict)

    def test_keys_match_available_columns(self):
        df     = make_full_df()
        result = build_all_narratives(df)
        for key in result:
            assert key in df.columns

    def test_all_values_are_non_empty_lists(self):
        df     = make_full_df()
        result = build_all_narratives(df)
        for key, findings in result.items():
            assert isinstance(findings, list), f"Findings for '{key}' is not a list."
            assert len(findings) > 0, f"Findings for '{key}' is empty."
