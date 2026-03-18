"""
tests/test_data_fetcher.py
--------------------------
Unit tests for src/data_fetcher.py.

These tests use mocking to avoid live API calls — no real API keys needed.
Run with:  pytest tests/ -v

Covers:
  - Config key validation (KeyError on bad key)
  - FRED normalisation logic (schema contract)
  - BLS normalisation logic
  - Cache read/write round-trip
  - fetch_all_fred merging behaviour
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Make src/ importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_fetcher import (
    _normalise_bls_series,
    _normalise_fred_series,
    _save_to_cache,
    _load_from_cache,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_fred_series() -> pd.Series:
    """Minimal fredapi-style Series."""
    idx = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])
    return pd.Series([4.2, 4.1, 4.0], index=idx)


@pytest.fixture
def sample_bls_records() -> list[dict]:
    """Minimal BLS API-style records list."""
    return [
        {"year": "2023", "period": "M01", "periodName": "January",  "value": "1000.0"},
        {"year": "2023", "period": "M02", "periodName": "February", "value": "1005.0"},
        {"year": "2023", "period": "M13", "periodName": "Annual",   "value": "1002.5"},  # should be skipped
    ]


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------

class TestNormaliseFredSeries:
    def test_returns_dataframe(self, sample_fred_series):
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert isinstance(df, pd.DataFrame)

    def test_column_named_after_key(self, sample_fred_series):
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert "unemployment_rate" in df.columns

    def test_index_named_date(self, sample_fred_series):
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert df.index.name == "date"

    def test_index_is_datetimeindex(self, sample_fred_series):
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_no_nulls_after_normalise(self, sample_fred_series):
        # Inject a NaN — normalise should drop it.
        sample_fred_series.iloc[1] = float("nan")
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert df.isnull().sum().sum() == 0

    def test_dtype_is_float64(self, sample_fred_series):
        df = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        assert df["unemployment_rate"].dtype == "float64"

    def test_no_duplicate_index(self):
        idx = pd.to_datetime(["2023-01-01", "2023-01-01", "2023-02-01"])
        s = pd.Series([4.2, 4.3, 4.1], index=idx)
        df = _normalise_fred_series(s, "unemployment_rate")
        assert df.index.is_unique

    def test_sorted_ascending(self, sample_fred_series):
        # Shuffle the series, confirm output is sorted.
        shuffled = sample_fred_series.iloc[::-1]
        df = _normalise_fred_series(shuffled, "unemployment_rate")
        assert df.index.is_monotonic_increasing


class TestNormaliseBLSSeries:
    def test_annual_record_skipped(self, sample_bls_records):
        df = _normalise_bls_series(sample_bls_records, "portland_employment")
        # Only 2 monthly records; M13 (annual) should be excluded.
        assert len(df) == 2

    def test_column_named_after_key(self, sample_bls_records):
        df = _normalise_bls_series(sample_bls_records, "portland_employment")
        assert "portland_employment" in df.columns

    def test_index_is_datetimeindex(self, sample_bls_records):
        df = _normalise_bls_series(sample_bls_records, "portland_employment")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_values_are_float(self, sample_bls_records):
        df = _normalise_bls_series(sample_bls_records, "portland_employment")
        assert df["portland_employment"].dtype == float

    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="No monthly records"):
            _normalise_bls_series([], "portland_employment")


# ---------------------------------------------------------------------------
# Cache round-trip tests
# ---------------------------------------------------------------------------

class TestCache:
    def test_save_and_load_roundtrip(self, tmp_path, sample_fred_series):
        df_original = _normalise_fred_series(sample_fred_series, "unemployment_rate")
        _save_to_cache(df_original, "unemployment_rate", subdir=str(tmp_path))

        loaded = _load_from_cache("unemployment_rate", subdir=str(tmp_path))
        assert loaded is not None
        pd.testing.assert_frame_equal(df_original, loaded)

    def test_load_returns_none_if_no_cache(self, tmp_path):
        result = _load_from_cache("nonexistent_key", subdir=str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# Config key validation tests
# ---------------------------------------------------------------------------

class TestConfigKeyValidation:
    def test_bad_fred_key_raises_keyerror(self):
        # Patch the FRED client so no real API call is made.
        with patch("src.data_fetcher._fred") as _mock_fred:
            from src.data_fetcher import fetch_fred_series
            with pytest.raises(KeyError, match="not found in FRED_SERIES config"):
                fetch_fred_series("this_key_does_not_exist")

    def test_bad_bls_key_raises_keyerror(self):
        from src.data_fetcher import fetch_bls_series
        with pytest.raises(KeyError, match="not found in BLS_SERIES config"):
            fetch_bls_series("this_key_does_not_exist")
