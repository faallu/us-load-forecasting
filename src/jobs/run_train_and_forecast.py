"""XGBoost-only multi-window train / forecast / backtest job.

For each forecast window (7d/14d/1mo hourly; 3mo/6mo/1yr daily) this:
  1. loads the window's tuned hyperparameters (running a grid search if none exist),
  2. produces a forward forecast at the window's resolution,
  3. runs a chronological train/test backtest,
  4. writes per-window forecast / metrics / plot artifacts to storage (local or S3).

The deployed pipeline is XGBoost-only; SARIMA remains in the repo but is not used here.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src import storage
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.models.backtest import run_backtest
from src.models.predict import build_load_forecast, save_forecast
from src.models.xgboost_model import XgbConfig
from src.windows import DEFAULT_WINDOW, ForecastWindow, all_windows, get_window

LOGGER = logging.getLogger(__name__)


def _resolve_windows(window_keys: list[str] | None) -> list[ForecastWindow]:
    if not window_keys:
        return all_windows()
    return [get_window(key) for key in window_keys]


def _processed_path(settings: Settings, resolution: str) -> str:
    return settings.processed_load_daily_path if resolution == "daily" else settings.processed_load_path


def _load_processed(settings: Settings, resolution: str, cache: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    """Load (and cache) the processed frame for a resolution.

    Hourly windows read ``load_hourly.parquet``; daily windows read ``load_daily.parquet``
    (pulled directly from the EIA daily endpoint).
    """
    path = _processed_path(settings, resolution)
    if resolution not in cache:
        if not storage.exists(path):
            raise FileNotFoundError(f"Processed {resolution} load data not found at {path}. Run ingestion first.")
        cache[resolution] = storage.read_parquet(path)
    return cache[resolution], path


def _config_for_window(
    settings: Settings,
    load_df: pd.DataFrame,
    window: ForecastWindow,
    tune_if_missing: bool,
) -> XgbConfig:
    from src.models.tuning import load_best_params, tune_and_save_window

    params_path = settings.best_params_path(window.key)
    best = load_best_params(params_path)
    if best is None and tune_if_missing:
        LOGGER.info("No tuned params for window=%s; running grid search.", window.key)
        best, _ = tune_and_save_window(load_df, window, params_path)
    if best is None:
        LOGGER.info("Using default params for window=%s (no tuning).", window.key)
        best = {}
    return XgbConfig.for_resolution(window.resolution, **best)


def run_window(
    settings: Settings,
    load_df: pd.DataFrame,
    actual_path: str,
    window: ForecastWindow,
    tune_if_missing: bool = True,
) -> None:
    LOGGER.info("=== Window %s (%s, horizon=%s) ===", window.key, window.resolution, window.horizon)
    config = _config_for_window(settings, load_df, window, tune_if_missing)

    load_fcst = build_load_forecast(
        load_df=load_df,
        horizon_hours=window.horizon,
        max_train_hours=None,  # XGBoost trains on all available history
        model="xgboost",
        xgb_config=config,
    )
    forecast_path = settings.forecast_path(window.key)
    save_forecast(load_fcst, forecast_path)
    # Keep the legacy non-windowed XGBoost path fed so the existing Streamlit app works.
    if window.key == DEFAULT_WINDOW:
        save_forecast(load_fcst, settings.forecast_load_xgb_path)
    LOGGER.info("Forecast complete: window=%s rows=%s -> %s", window.key, len(load_fcst), forecast_path)

    saved = run_backtest(
        actual_path=actual_path,
        kind="load",
        holdout_hours=window.horizon,
        metrics_txt_path=settings.backtest_metrics_path(window.key),
        forecast_output=settings.backtest_forecast_path(window.key),
        metrics_output=settings.backtest_metrics_parquet_path(window.key),
        plot_output=settings.backtest_plot_path(window.key),
        model="xgboost",
        xgb_config=config,
        test_fraction=settings.xgb_test_fraction,
        resolution=window.resolution,
        window_label=window.label,
    )
    LOGGER.info("Backtest complete: window=%s metrics=%s plot=%s", window.key, saved.get("metrics_txt"), saved.get("plot"))


def run_train_and_forecast(window_keys: list[str] | None = None, tune_if_missing: bool = True) -> None:
    setup_logging()
    settings = get_settings()
    windows = _resolve_windows(window_keys)
    LOGGER.info("Training XGBoost for windows: %s", [w.key for w in windows])
    cache: dict[str, pd.DataFrame] = {}
    for window in windows:
        load_df, actual_path = _load_processed(settings, window.resolution, cache)
        run_window(settings, load_df, actual_path, window, tune_if_missing=tune_if_missing)


def run_tuning(window_keys: list[str] | None = None) -> None:
    """Grid-search and persist best params for each window (no forecasting)."""
    from src.models.tuning import tune_and_save_window

    setup_logging()
    settings = get_settings()
    windows = _resolve_windows(window_keys)
    LOGGER.info("Tuning XGBoost for windows: %s", [w.key for w in windows])
    cache: dict[str, pd.DataFrame] = {}
    for window in windows:
        load_df, _ = _load_processed(settings, window.resolution, cache)
        tune_and_save_window(load_df, window, settings.best_params_path(window.key))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost multi-window train/forecast/backtest.")
    parser.add_argument(
        "--window",
        action="append",
        dest="windows",
        default=None,
        help="Window key to run (repeatable). Defaults to all windows.",
    )
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Only run the grid search and persist best params (no forecast/backtest).",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Do not grid-search when a window has no saved params; use model defaults.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.tune_only:
        run_tuning(args.windows)
    else:
        run_train_and_forecast(args.windows, tune_if_missing=not args.no_tune)


if __name__ == "__main__":
    main()
