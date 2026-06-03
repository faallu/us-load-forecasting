from __future__ import annotations

import os
from dataclasses import dataclass

from src import storage

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    eia_api_key: str
    eia_base_url: str
    eia_daily_timezone: str
    default_history_start: str
    pull_backfill_hours: int
    ingest_interval_minutes: int
    train_interval_hours: int
    forecast_horizon_hours: int
    max_train_hours: int
    forecast_model: str
    xgb_n_estimators: int
    xgb_max_depth: int
    xgb_learning_rate: float
    xgb_subsample: float
    xgb_colsample_bytree: float
    xgb_min_child_weight: int
    xgb_test_fraction: float
    storage_backend: str
    data_root: str
    processed_root: str
    forecasts_root: str
    state_root: str
    metrics_root: str
    reports_root: str
    models_root: str
    state_path: str
    processed_load_path: str
    processed_load_daily_path: str
    # Legacy single-window paths (kept so the SARIMA tooling and Streamlit apps work).
    forecast_load_path: str
    backtest_load_forecast_path: str
    backtest_load_metrics_path: str
    backtest_load_plot_path: str
    forecast_load_xgb_path: str
    backtest_load_forecast_xgb_path: str
    backtest_load_metrics_xgb_path: str
    backtest_load_plot_xgb_path: str
    allowed_regions: tuple[str, ...]

    # --- Per-window XGBoost artifact paths -------------------------------------
    def forecast_path(self, window: str) -> str:
        return storage.join(self.forecasts_root, f"load_forecast_xgb_{window}.parquet")

    def backtest_forecast_path(self, window: str) -> str:
        return storage.join(self.forecasts_root, f"load_backtest_forecast_xgb_{window}.parquet")

    def backtest_metrics_path(self, window: str) -> str:
        return storage.join(self.metrics_root, f"load_backtest_metrics_xgb_{window}.txt")

    def backtest_metrics_parquet_path(self, window: str) -> str:
        return storage.join(self.metrics_root, f"load_backtest_metrics_xgb_{window}.parquet")

    def backtest_plot_path(self, window: str) -> str:
        return storage.join(self.reports_root, f"load_backtest_xgb_{window}.png")

    def best_params_path(self, window: str) -> str:
        return storage.join(self.models_root, f"{window}.json")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name, str(default))
    return float(value)


def _resolve_data_root() -> tuple[str, str]:
    """Return ``(storage_backend, data_root)`` based on environment variables."""
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend == "s3":
        bucket = os.getenv("S3_BUCKET", "").strip()
        if not bucket:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND=s3.")
        prefix = os.getenv("S3_PREFIX", "").strip().strip("/")
        root = f"s3://{bucket}"
        if prefix:
            root = f"{root}/{prefix}"
        return backend, root
    return "local", os.getenv("DATA_ROOT", "data").strip()


def get_settings() -> Settings:
    storage_backend, data_root = _resolve_data_root()

    processed_root = storage.join(data_root, "processed")
    forecasts_root = storage.join(data_root, "forecasts")
    state_root = storage.join(data_root, "state")
    metrics_root = storage.join(data_root, "metrics")
    reports_root = storage.join(data_root, "reports")
    models_root = storage.join(data_root, "models", "best_params")

    return Settings(
        eia_api_key=os.getenv("EIA_API_KEY", ""),
        eia_base_url=os.getenv("EIA_BASE_URL", "https://api.eia.gov/v2"),
        eia_daily_timezone=os.getenv("EIA_DAILY_TIMEZONE", "Eastern").strip(),
        default_history_start=os.getenv("DEFAULT_HISTORY_START", "2023-01-01T00:00:00Z"),
        pull_backfill_hours=_int_env("PULL_BACKFILL_HOURS", 24),
        ingest_interval_minutes=_int_env("INGEST_INTERVAL_MINUTES", 60),
        train_interval_hours=_int_env("TRAIN_INTERVAL_HOURS", 24),
        forecast_horizon_hours=_int_env("FORECAST_HORIZON_HOURS", 168),
        max_train_hours=_int_env("MAX_TRAIN_HOURS", 2160),
        forecast_model=os.getenv("FORECAST_MODEL", "xgboost").strip().lower(),
        xgb_n_estimators=_int_env("XGB_N_ESTIMATORS", 600),
        xgb_max_depth=_int_env("XGB_MAX_DEPTH", 6),
        xgb_learning_rate=_float_env("XGB_LEARNING_RATE", 0.03),
        xgb_subsample=_float_env("XGB_SUBSAMPLE", 0.9),
        xgb_colsample_bytree=_float_env("XGB_COLSAMPLE_BYTREE", 0.9),
        xgb_min_child_weight=_int_env("XGB_MIN_CHILD_WEIGHT", 1),
        xgb_test_fraction=_float_env("XGB_TEST_FRACTION", 0.2),
        storage_backend=storage_backend,
        data_root=data_root,
        processed_root=processed_root,
        forecasts_root=forecasts_root,
        state_root=state_root,
        metrics_root=metrics_root,
        reports_root=reports_root,
        models_root=models_root,
        state_path=storage.join(state_root, "ingestion_state.json"),
        processed_load_path=storage.join(processed_root, "load_hourly.parquet"),
        processed_load_daily_path=storage.join(processed_root, "load_daily.parquet"),
        forecast_load_path=storage.join(forecasts_root, "load_forecast.parquet"),
        backtest_load_forecast_path=storage.join(forecasts_root, "load_backtest_forecast.parquet"),
        backtest_load_metrics_path=storage.join(metrics_root, "load_backtest_metrics_168h.txt"),
        backtest_load_plot_path=storage.join(reports_root, "load_backtest_168h.png"),
        forecast_load_xgb_path=storage.join(forecasts_root, "load_forecast_xgb.parquet"),
        backtest_load_forecast_xgb_path=storage.join(forecasts_root, "load_backtest_forecast_xgb.parquet"),
        backtest_load_metrics_xgb_path=storage.join(metrics_root, "load_backtest_metrics_xgb_168h.txt"),
        backtest_load_plot_xgb_path=storage.join(reports_root, "load_backtest_xgb_168h.png"),
        allowed_regions=("CISO", "ERCO", "ISNE", "MISO", "NYIS", "PJM", "SWPP"),
    )
