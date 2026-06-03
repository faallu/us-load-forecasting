from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src import storage
from src.config import get_settings
from src.logging_config import setup_logging
from src.models.evaluate import evaluate_forecast, evaluate_load_backtest

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Load forecast evaluation (overlap or backtest).")
    p.add_argument(
        "--backtest-hours",
        type=int,
        default=0,
        help="If > 0, hold out this many trailing hours per series and refit (recommended when saved forecasts do not overlap actuals).",
    )
    p.add_argument(
        "--overlap",
        action="store_true",
        help="Join saved load forecast parquet to processed actuals (needs actuals through forecast target times).",
    )
    p.add_argument(
        "--per-series-out",
        type=Path,
        default=None,
        help="Optional parquet path for per-series metrics.",
    )
    p.add_argument(
        "--max-train-days",
        type=int,
        default=0,
        help="If > 0, drop rows older than this many days before the global max period (much faster SARIMAX on long histories).",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    settings = get_settings()
    load_actual = storage.read_parquet(settings.processed_load_path)
    load_actual["period"] = pd.to_datetime(load_actual["period"], utc=True)
    if args.max_train_days > 0:
        cutoff = load_actual["period"].max() - pd.Timedelta(days=args.max_train_days)
        load_actual = load_actual.loc[load_actual["period"] >= cutoff].copy()
        LOGGER.info("Trimmed load history to last %s days (rows=%s)", args.max_train_days, len(load_actual))

    if args.backtest_hours > 0:
        by_series, overall = evaluate_load_backtest(load_actual, holdout_hours=args.backtest_hours)
        print("Load backtest (trailing holdout, SARIMAX refit per series)")
        print(f"holdout_hours={args.backtest_hours}")
    elif args.overlap:
        if not storage.exists(settings.forecast_load_path):
            raise FileNotFoundError(f"Missing forecast file: {settings.forecast_load_path}")
        load_fcst = storage.read_parquet(settings.forecast_load_path)
        by_series, overall = evaluate_forecast(
            load_actual,
            load_fcst,
            group_cols=["region", "series"],
        )
        print("Load overlap evaluation (forecast target_time vs actual period)")
    else:
        raise SystemExit("Specify --backtest-hours N and/or --overlap (see --help).")

    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nPer-series (all):")
    print(by_series.to_string(index=False))

    if args.per_series_out is not None:
        args.per_series_out.parent.mkdir(parents=True, exist_ok=True)
        by_series.to_parquet(args.per_series_out, index=False)
        LOGGER.info("Wrote per-series metrics to %s", args.per_series_out)


if __name__ == "__main__":
    main()
