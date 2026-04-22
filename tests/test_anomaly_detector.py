"""
tests/test_anomaly_detector.py
-------------------------------
Unit tests for src/anomaly_detector.py.

Uses synthetic data so no API key or network access is required.

Run with:
    pytest tests/test_anomaly_detector.py -v
"""

import pytest
import pandas as pd
import numpy as np

from src.anomaly_detector import detect_anomalies, detect_all_anomalies, get_anomaly_events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_flat_series(n: int = 60, value: float = 5.0, key: str = "unemployment_rate") -> pd.DataFrame:
    """Completely flat series — no anomalies possible."""
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.DataFrame({key: [value] * n}, index=idx)


def make_series_with_spike(
    n: int = 60,
    spike_pos: int = 40,
    spike_magnitude: float = 10.0,
    key: str = "unemployment_rate",
) -> pd.DataFrame:
    """Flat series with one large positive spike at ``spike_pos``."""
    idx    = pd.date_range("2019-01-01", periods=n, freq="MS")
    values = [5.0] * n
    values[spike_pos] = 5.0 + spike_magnitude
    return pd.DataFrame({key: values}, index=idx)


def make_noisy_series(n: int = 100, key: str = "cpi_west") -> pd.DataFrame:
    """Random-walk series with low noise — most points should not be flagged."""
    rng    = np.random.default_rng(99)
    idx    = pd.date_range("2015-01-01", periods=n, freq="MS")
    values = 250.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({key: values}, index=idx)


def make_multi_key_df(n: int = 60) -> pd.DataFrame:
    """Synthetic DataFrame with multiple configured keys."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "unemployment_rate": 5.0 + rng.normal(0, 0.3, n),
            "cpi_west":          250.0 + rng.normal(0, 1.0, n),
            "fed_funds_rate":    2.5 + rng.normal(0, 0.2, n),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# detect_anomalies — return type and schema
# ---------------------------------------------------------------------------

class TestDetectAnomaliesSchema:
    def test_returns_boolean_series(self):
        df     = make_noisy_series()
        result = detect_anomalies(df, "cpi_west")
        assert isinstance(result, pd.Series)
        assert result.dtype == bool

    def test_index_matches_non_null_series(self):
        df     = make_noisy_series()
        result = detect_anomalies(df, "cpi_west")
        expected_index = df["cpi_west"].dropna().index
        pd.testing.assert_index_equal(result.index, expected_index)

    def test_series_name(self):
        df     = make_noisy_series()
        result = detect_anomalies(df, "cpi_west")
        assert result.name == "cpi_west_anomaly"

    def test_no_nulls_in_result(self):
        df     = make_noisy_series()
        result = detect_anomalies(df, "cpi_west")
        assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# detect_anomalies — detection correctness
# ---------------------------------------------------------------------------

class TestDetectAnomaliesCorrectness:
    def test_flat_series_no_anomalies(self):
        """A perfectly flat series has zero variance → no anomalies."""
        df     = make_flat_series()
        result = detect_anomalies(df, "unemployment_rate")
        assert result.sum() == 0

    def test_large_spike_is_flagged(self):
        """A spike of 10× the rolling std should be flagged."""
        df     = make_series_with_spike(spike_magnitude=10.0)
        result = detect_anomalies(df, "unemployment_rate")
        assert result.sum() >= 1

    def test_spike_date_is_flagged(self):
        """The specific spike index should be among the flagged dates."""
        n          = 60
        spike_pos  = 40
        df         = make_series_with_spike(n=n, spike_pos=spike_pos)
        result     = detect_anomalies(df, "unemployment_rate")
        spike_date = df.index[spike_pos]
        assert result.loc[spike_date] is True  # noqa: E712 (intentional boolean check)

    def test_custom_threshold_changes_count(self):
        """A lower threshold should flag more points than a higher one."""
        df    = make_noisy_series()
        low   = detect_anomalies(df, "cpi_west", threshold=1.0)
        high  = detect_anomalies(df, "cpi_west", threshold=3.0)
        assert low.sum() >= high.sum()

    def test_missing_column_raises(self):
        df = make_noisy_series()
        with pytest.raises(KeyError, match="not found in DataFrame"):
            detect_anomalies(df, "nonexistent_column")


# ---------------------------------------------------------------------------
# detect_all_anomalies
# ---------------------------------------------------------------------------

class TestDetectAllAnomalies:
    def test_returns_dict(self):
        df     = make_multi_key_df()
        result = detect_all_anomalies(df)
        assert isinstance(result, dict)

    def test_keys_match_available_columns(self):
        df     = make_multi_key_df()
        result = detect_all_anomalies(df)
        for key in result:
            assert key in df.columns

    def test_missing_series_skipped(self):
        """home_price_index not in fixture — should not appear in results."""
        df     = make_multi_key_df()
        result = detect_all_anomalies(df)
        assert "home_price_index" not in result

    def test_all_values_boolean_series(self):
        df     = make_multi_key_df()
        result = detect_all_anomalies(df)
        for key, flags in result.items():
            assert flags.dtype == bool, f"Flags for '{key}' are not boolean."


# ---------------------------------------------------------------------------
# get_anomaly_events
# ---------------------------------------------------------------------------

class TestGetAnomalyEvents:
    def test_empty_flags_returns_empty_df(self):
        df   = make_noisy_series()
        key  = "cpi_west"
        flags = detect_anomalies(df, key)
        flags[:] = False  # force no anomalies
        result = get_anomaly_events(df, key, flags)
        assert len(result) == 0
        assert set(result.columns) == {"start", "end", "peak_value", "direction", "duration_months"}

    def test_spike_produces_one_event(self):
        df    = make_series_with_spike(spike_pos=40, spike_magnitude=10.0)
        flags = detect_anomalies(df, "unemployment_rate")
        result = get_anomaly_events(df, "unemployment_rate", flags)
        assert len(result) >= 1

    def test_event_direction_above(self):
        """A positive spike should produce direction='above'."""
        df     = make_series_with_spike(spike_magnitude=10.0)
        flags  = detect_anomalies(df, "unemployment_rate")
        result = get_anomaly_events(df, "unemployment_rate", flags)
        assert "above" in result["direction"].values

    def test_n_recent_limits_rows(self):
        """n_recent should cap the number of returned events."""
        df    = make_series_with_spike(spike_pos=20, spike_magnitude=10.0)
        # Add a second spike
        df.iloc[50, 0] = 30.0
        flags = detect_anomalies(df, "unemployment_rate")
        result = get_anomaly_events(df, "unemployment_rate", flags, n_recent=1)
        assert len(result) <= 1

    def test_event_start_before_end(self):
        df     = make_series_with_spike()
        flags  = detect_anomalies(df, "unemployment_rate")
        result = get_anomaly_events(df, "unemployment_rate", flags)
        for _, row in result.iterrows():
            assert row["start"] <= row["end"]
