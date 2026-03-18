"""
run_pipeline.py
---------------
Phase 1 pipeline entry point for the PNW Regional Economic Health Dashboard.

Run this script to:
  1. Pull all configured FRED series from the API
  2. Save individual raw CSVs to data/raw/
  3. Merge into a single combined CSV at data/processed/all_indicators.csv

Usage
-----
    # From the project root (with venv activated):
    python run_pipeline.py

    # Force a fresh API pull even if cache exists:
    python run_pipeline.py --no-cache

Flags
-----
    --no-cache      Ignore any existing cached CSVs and re-pull from APIs.
    --start YYYY-MM-DD   Override the default start date from config.
    --end   YYYY-MM-DD   Override the default end date (default: today).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_END_DATE, DEFAULT_START_DATE, FRED_SERIES
from src.data_fetcher import fetch_all_fred, save_processed

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PNW Economic Dashboard — Phase 1 data pipeline"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-pull all data from APIs, ignoring local cache.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        metavar="YYYY-MM-DD",
        help=f"Start date for data pull (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
        metavar="YYYY-MM-DD",
        help="End date for data pull (default: today)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cache = not args.no_cache

    logger.info("=" * 60)
    logger.info("PNW Dashboard — Phase 1 Pipeline")
    logger.info("  Date range : %s → %s", args.start, args.end or "today")
    logger.info("  Cache mode : %s", "ON" if use_cache else "OFF (fresh pull)")
    logger.info("  Series     : %s", list(FRED_SERIES.keys()))
    logger.info("=" * 60)

    # --- Fetch all FRED series ---
    df_all = fetch_all_fred(
        start_date=args.start,
        end_date=args.end,
        use_cache=use_cache,
    )

    # --- Basic sanity checks ---
    logger.info("\nData shape  : %s", df_all.shape)
    logger.info("Date range  : %s → %s", df_all.index.min().date(), df_all.index.max().date())
    logger.info("Missing vals:\n%s", df_all.isnull().sum().to_string())

    # --- Save combined processed output ---
    out_path = save_processed(df_all, "all_indicators.csv")
    logger.info("\n✓ Pipeline complete. Processed data saved to: %s", out_path)

    # --- Print preview ---
    print("\n--- Preview (last 5 rows) ---")
    print(df_all.tail())


if __name__ == "__main__":
    main()
