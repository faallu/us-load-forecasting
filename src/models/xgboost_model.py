"""XGBoost load-forecasting model (hourly and daily resolutions).

Mirrors the public surface of ``src/models/sarima.py`` so the rest of the pipeline
(``predict.py`` / ``backtest.py``) can swap models without other changes. Because
XGBoost is not a native time-series model, each series is turned into a supervised
problem with lag / calendar / rolling features, and multi-step forecasts are produced
recursively (each prediction is fed back in as the next step's lag).

Short windows (7d/14d/1mo) use hourly resolution; long windows (3mo/6mo/1yr) use
daily resolution. The resolution is carried on ``XgbConfig`` and drives the resample
frequency, the calendar features, the lag set, and the recursive step size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

LOGGER = logging.getLogger(__name__)

HOURLY_CALENDAR_COLS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

DAILY_CALENDAR_COLS = [
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

DEFAULT_HOURLY_LAGS = (1, 2, 3, 24, 25, 48, 168)
DEFAULT_HOURLY_ROLL = (24, 168)
DEFAULT_DAILY_LAGS = (1, 2, 3, 7, 14, 28, 365)
DEFAULT_DAILY_ROLL = (7, 30)


@dataclass(frozen=True)
class XgbConfig:
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: int = 1
    resolution: str = "hourly"
    lags: tuple[int, ...] = DEFAULT_HOURLY_LAGS
    roll_windows: tuple[int, ...] = DEFAULT_HOURLY_ROLL
    min_observations: int = 96
    random_state: int = 42

    @property
    def freq(self) -> str:
        return "h" if self.resolution == "hourly" else "D"

    @property
    def step(self) -> pd.Timedelta:
        return pd.Timedelta(hours=1) if self.resolution == "hourly" else pd.Timedelta(days=1)

    @property
    def calendar_cols(self) -> list[str]:
        return HOURLY_CALENDAR_COLS if self.resolution == "hourly" else DAILY_CALENDAR_COLS

    @property
    def feature_cols(self) -> list[str]:
        lag_cols = [f"lag_{lag}" for lag in self.lags]
        roll_cols = [f"rollmean_{w}" for w in self.roll_windows]
        return self.calendar_cols + lag_cols + roll_cols

    def to_xgb_params(self) -> dict[str, object]:
        return {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
        }

    @classmethod
    def for_resolution(cls, resolution: str, **overrides: object) -> "XgbConfig":
        """Build a config with sensible feature defaults for the given resolution.

        ``overrides`` may carry tuned hyperparameters (n_estimators, max_depth, ...);
        unknown keys are ignored so a best-params JSON can be splatted in directly.
        """
        if resolution == "daily":
            defaults: dict[str, object] = {
                "resolution": "daily",
                "lags": DEFAULT_DAILY_LAGS,
                "roll_windows": DEFAULT_DAILY_ROLL,
                "min_observations": 60,
            }
        else:
            defaults = {
                "resolution": "hourly",
                "lags": DEFAULT_HOURLY_LAGS,
                "roll_windows": DEFAULT_HOURLY_ROLL,
                "min_observations": 96,
            }
        allowed = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "random_state",
            "lags",
            "roll_windows",
            "min_observations",
        }
        defaults.update({k: v for k, v in overrides.items() if k in allowed})
        return cls(**defaults)


def _prepare_series(series: pd.Series, config: XgbConfig) -> pd.Series:
    """Align a series to the config's resolution frequency.

    Hourly windows read the hourly demand series; daily windows read the daily demand
    series pulled directly from EIA (already one value per day), so this only enforces a
    regular frequency and never aggregates.
    """
    clean = series.sort_index().astype(float)
    if config.resolution == "daily":
        return clean.asfreq("D")
    return clean.asfreq("h")


def _build_features(ts: pd.Series, config: XgbConfig) -> pd.DataFrame:
    """Build a supervised feature frame from a series indexed by timestamp."""
    df = pd.DataFrame({"y": ts.astype(float)})
    idx = df.index
    if config.resolution == "hourly":
        df["hour"] = idx.hour
        df["dayofweek"] = idx.dayofweek
        df["month"] = idx.month
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    else:
        df["dayofweek"] = idx.dayofweek
        df["month"] = idx.month
        df["dayofyear"] = idx.dayofyear
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    for lag in config.lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for window in config.roll_windows:
        df[f"rollmean_{window}"] = df["y"].shift(1).rolling(window).mean()
    return df


def _recursive_forecast(
    model: XGBRegressor,
    clean: pd.Series,
    horizon: int,
    config: XgbConfig,
) -> list[float]:
    feature_cols = config.feature_cols
    # Only the most recent observations are needed to compute the next row's features.
    tail_len = max(max(config.lags, default=1), max(config.roll_windows, default=1)) + 2
    step = config.step

    history = clean.copy()
    preds: list[float] = []
    for _ in range(horizon):
        next_time = history.index[-1] + step
        window = history.iloc[-tail_len:].copy()
        window.loc[next_time] = np.nan
        feature_row = _build_features(window, config).iloc[[-1]][feature_cols]
        yhat = float(model.predict(feature_row)[0])
        history.loc[next_time] = yhat
        preds.append(yhat)
    return preds


def fit_and_forecast_series_with_info(
    series: pd.Series,
    horizon: int,
    config: XgbConfig | None = None,
) -> tuple[pd.Series, dict[str, float | None]]:
    """Point forecast plus lightweight training info from the fitted XGBoost model.

    Falls back to last-value persistence on series that are too short to train, matching
    the behaviour of ``sarima.fit_and_forecast_series_with_info``.
    """
    config = config or XgbConfig()
    clean = _prepare_series(series, config)
    if clean.isna().all():
        return pd.Series([0.0] * horizon), {}
    clean = clean.interpolate(limit_direction="both").ffill().bfill()
    if len(clean) < config.min_observations:
        if clean.empty:
            return pd.Series([0.0] * horizon), {}
        last = float(clean.iloc[-1])
        return pd.Series([last] * horizon), {}

    features = _build_features(clean, config).dropna()
    if len(features) < 50:
        last = float(clean.iloc[-1])
        return pd.Series([last] * horizon), {}

    x_train = features[config.feature_cols]
    y_train = features["y"]

    model = XGBRegressor(**config.to_xgb_params())
    model.fit(x_train, y_train)

    preds = _recursive_forecast(model, clean, horizon, config)
    info = {
        "train_rows": float(len(features)),
        "n_features": float(len(config.feature_cols)),
    }
    return pd.Series(preds).astype(float).reset_index(drop=True), info


def fit_and_forecast_series(
    series: pd.Series,
    horizon: int,
    config: XgbConfig | None = None,
) -> pd.Series:
    forecast, _ = fit_and_forecast_series_with_info(series, horizon, config=config)
    return forecast


def build_split_forecast(
    actual_df: pd.DataFrame,
    group_cols: list[str],
    test_fraction: float,
    value_col: str = "value",
    config: XgbConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split evaluation per series.

    For each series the engineered feature rows are split in time: the first
    ``1 - test_fraction`` fraction trains the model and the final ``test_fraction``
    is predicted in a single batch using its *true* lagged features (lag_1 is the
    actual previous step, etc.). This is the standard supervised evaluation for a
    tabular model and, unlike a recursive multi-step forecast, stays meaningful and
    fast over a large (e.g. 20%) test window.

    Resolution-aware: when ``config.resolution == "daily"`` the series is resampled
    to daily means first, so ``issue_time``/``target_time`` land on daily timestamps.

    Returns ``(forecast_df, info_df)`` with the same schema the backtest plumbing
    expects: forecast rows carry ``issue_time`` / ``target_time`` / ``yhat`` and the
    info frame carries per-series ``train_rows`` / ``test_rows``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1 (exclusive)")

    config = config or XgbConfig()
    feature_cols = config.feature_cols
    local = actual_df.copy()
    local["period"] = pd.to_datetime(local["period"], utc=True)
    local[value_col] = pd.to_numeric(local[value_col], errors="coerce")

    forecast_rows: list[dict[str, object]] = []
    info_rows: list[dict[str, object]] = []

    for group_values, gdf in local.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        ts = gdf.set_index("period")[value_col]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index()
        clean = _prepare_series(ts, config)
        if clean.isna().all():
            continue
        clean = clean.interpolate(limit_direction="both").ffill().bfill()

        features = _build_features(clean, config).dropna()
        n = len(features)
        if n < config.min_observations:
            continue
        n_test = int(round(n * test_fraction))
        n_train = n - n_test
        if n_test < 1 or n_train < 50:
            continue

        train = features.iloc[:n_train]
        test = features.iloc[n_train:]

        model = XGBRegressor(**config.to_xgb_params())
        model.fit(train[feature_cols], train["y"])
        preds = model.predict(test[feature_cols])

        issue_time = train.index[-1]
        base_row = {col: group_values[idx] for idx, col in enumerate(group_cols)}
        for target_time, yhat in zip(test.index, preds):
            forecast_rows.append(
                {
                    **base_row,
                    "issue_time": issue_time,
                    "target_time": target_time,
                    "yhat": float(yhat),
                }
            )
        info_rows.append(
            {
                **base_row,
                "issue_time": issue_time,
                "train_rows": int(n_train),
                "test_rows": int(n_test),
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    if forecast_df.empty:
        forecast_df = pd.DataFrame(columns=[*group_cols, "issue_time", "target_time", "yhat"])
    else:
        forecast_df = forecast_df.sort_values([*group_cols, "target_time"]).reset_index(drop=True)

    info_df = pd.DataFrame(info_rows)
    if info_df.empty:
        info_df = pd.DataFrame(columns=[*group_cols, "issue_time", "train_rows", "test_rows"])
    else:
        info_df = info_df.sort_values(group_cols).reset_index(drop=True)

    return forecast_df, info_df


def forecast_grouped_dataframe(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    horizon: int,
    issue_time: pd.Timestamp | None = None,
    config: XgbConfig | None = None,
) -> pd.DataFrame:
    config = config or XgbConfig()
    if df.empty:
        cols = [*group_cols, "issue_time", "target_time", "yhat"]
        return pd.DataFrame(columns=cols)

    out_rows: list[dict[str, object]] = []
    local = df.copy()
    local["period"] = pd.to_datetime(local["period"], utc=True)
    step = config.step

    for group_values, group_df in local.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        ts = group_df.set_index("period")[value_col]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index()
        clean = _prepare_series(ts, config)
        non_null_ts = clean.dropna()
        if non_null_ts.empty:
            continue
        preds, _ = fit_and_forecast_series_with_info(ts, horizon=horizon, config=config)
        forecast_origin = non_null_ts.index.max()
        start_time = forecast_origin + step

        for offset, yhat in enumerate(preds):
            row = {col: group_values[idx] for idx, col in enumerate(group_cols)}
            row["issue_time"] = forecast_origin if issue_time is None else issue_time
            row["target_time"] = start_time + step * offset
            row["yhat"] = float(yhat)
            out_rows.append(row)

    if not out_rows:
        return pd.DataFrame(columns=[*group_cols, "issue_time", "target_time", "yhat"])
    return pd.DataFrame(out_rows).sort_values([*group_cols, "target_time"]).reset_index(drop=True)
