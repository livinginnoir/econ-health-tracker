"""
tests/test_forecaster.py
------------------------
Unit tests for src/forecaster.py.

These tests use synthetic data so they run without a FRED API key and without
requiring the full Prophet model to converge on real economic series.  Each
test targets a specific contract from the module's docstring.

Run with:
    pytest tests/test_forecaster.py -v
"""

import pytest
import pandas as pd
import numpy as np

from src.forecaster import forecast_series, forecast_all, _to_prophet_df, _from_prophet_forecast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_monthly_series(n_months: int = 60, start: str = "2019-01-01") -> pd.DataFrame:
    """Synthetic monthly series with a gentle upward trend + noise."""
    idx    = pd.date_range(start=start, periods=n_months, freq="MS")
    values = np.linspace(5.0, 8.0, n_months) + np.random.default_rng(42).normal(0, 0.3, n_months)
    return pd.DataFrame({"unemployment_rate": values}, index=idx)


def make_quarterly_series(n_quarters: int = 40, start: str = "2014-01-01") -> pd.DataFrame:
    """Synthetic quarterly series (home price index proxy)."""
    idx    = pd.date_range(start=start, periods=n_quarters, freq="QS")
    values = np.linspace(100.0, 220.0, n_quarters) + np.random.default_rng(7).normal(0, 3, n_quarters)
    return pd.DataFrame({"home_price_index": values}, index=idx)


def make_full_df(n_months: int = 60) -> pd.DataFrame:
    """
    Synthetic multi-column DataFrame matching the structure of fetch_all_fred().
    Only includes columns present in config.FRED_SERIES.
    """
    idx = pd.date_range(start="2019-01-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "unemployment_rate": np.linspace(5, 4, n_months) + rng.normal(0, 0.2, n_months),
            "cpi_west":          np.linspace(250, 310, n_months) + rng.normal(0, 1, n_months),
            "fed_funds_rate":    np.linspace(0.25, 5.0, n_months) + rng.normal(0, 0.1, n_months),
            # home_price_index intentionally omitted to test missing-column handling
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# _to_prophet_df
# ---------------------------------------------------------------------------

class TestToProphetDf:
    def test_output_columns(self):
        series = make_monthly_series()["unemployment_rate"]
        result = _to_prophet_df(series)
        assert list(result.columns) == ["ds", "y"]

    def test_no_timezone(self):
        series = make_monthly_series()["unemployment_rate"]
        result = _to_prophet_df(series)
        # Prophet fails on tz-aware timestamps
        assert result["ds"].dt.tz is None

    def test_length_preserved(self):
        series = make_monthly_series(n_months=36)["unemployment_rate"]
        result = _to_prophet_df(series)
        assert len(result) == 36


# ---------------------------------------------------------------------------
# _from_prophet_forecast
# ---------------------------------------------------------------------------

class TestFromProphetForecast:
    def _dummy_forecast(self, n: int = 80) -> pd.DataFrame:
        dates = pd.date_range("2019-01-01", periods=n, freq="MS")
        return pd.DataFrame({
            "ds":          dates,
            "yhat":        np.linspace(5, 8, n),
            "yhat_lower":  np.linspace(5, 8, n) - 1.5,
            "yhat_upper":  np.linspace(5, 8, n) + 1.5,
            "trend":       np.linspace(5, 8, n),   # extra column Prophet always adds
        })

    def test_correct_columns_returned(self):
        result = _from_prophet_forecast(self._dummy_forecast())
        assert set(result.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}

    def test_floor_clamps_lower_bound(self):
        """yhat_lower values below floor must be clamped."""
        df = self._dummy_forecast()
        df["yhat_lower"] = -5.0  # force all values below zero
        result = _from_prophet_forecast(df, floor=0.0)
        assert (result["yhat_lower"] >= 0.0).all()

    def test_negative_floor_allowed(self):
        """A floor of -inf means no clamping."""
        df = self._dummy_forecast()
        df["yhat_lower"] = -5.0
        result = _from_prophet_forecast(df, floor=float("-inf"))
        assert (result["yhat_lower"] < 0).any()

    def test_integer_index(self):
        """Result should have a plain integer index, not the original datetime index."""
        result = _from_prophet_forecast(self._dummy_forecast())
        assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# forecast_series
# ---------------------------------------------------------------------------

class TestForecastSeries:
    def test_returns_required_columns(self):
        df     = make_monthly_series()
        result = forecast_series(df, "unemployment_rate", horizon_months=6)
        assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(result.columns)

    def test_forecast_extends_beyond_last_date(self):
        df         = make_monthly_series()
        last_date  = df.index[-1]
        result     = forecast_series(df, "unemployment_rate", horizon_months=12)
        assert result["ds"].max() > last_date

    def test_forecast_horizon_length(self):
        """Future rows (beyond last historical date) should equal horizon_months."""
        df        = make_monthly_series()
        last_date = df.index[-1]
        result    = forecast_series(df, "unemployment_rate", horizon_months=12)
        future    = result[result["ds"] > last_date]
        assert len(future) == 12

    def test_quarterly_series(self):
        """Quarterly series should produce a quarterly forecast."""
        df        = make_quarterly_series()
        last_date = df.index[-1]
        result    = forecast_series(df, "home_price_index", horizon_months=12)
        future    = result[result["ds"] > last_date]
        # 12 months // 3 = 4 quarterly periods
        assert len(future) == 4

    def test_yhat_lower_non_negative(self):
        """value_floor=0.0 in config → yhat_lower should never be negative."""
        df     = make_monthly_series()
        result = forecast_series(df, "unemployment_rate", horizon_months=12)
        assert (result["yhat_lower"] >= 0).all()

    def test_invalid_key_raises(self):
        df = make_monthly_series()
        with pytest.raises(KeyError, match="not found in FRED_SERIES"):
            forecast_series(df, "nonexistent_key")

    def test_too_short_series_raises(self):
        """Fewer than 24 observations should raise ValueError."""
        idx    = pd.date_range("2023-01-01", periods=10, freq="MS")
        df     = pd.DataFrame({"unemployment_rate": [5.0] * 10}, index=idx)
        with pytest.raises(ValueError, match="At least 24"):
            forecast_series(df, "unemployment_rate")


# ---------------------------------------------------------------------------
# forecast_all
# ---------------------------------------------------------------------------

class TestForecastAll:
    def test_returns_dict_with_available_keys(self):
        df     = make_full_df()
        result = forecast_all(df, horizon_months=6)
        # Should contain keys that are present in df AND in FRED_SERIES
        for key in result:
            assert key in df.columns

    def test_missing_key_skipped_gracefully(self):
        """home_price_index is not in the fixture — should not raise, just be absent."""
        df     = make_full_df()
        result = forecast_all(df, horizon_months=6)
        assert "home_price_index" not in result

    def test_all_results_have_forecast_schema(self):
        df     = make_full_df()
        result = forecast_all(df, horizon_months=6)
        for key, fdf in result.items():
            assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(fdf.columns), (
                f"Forecast for '{key}' is missing required columns."
            )
