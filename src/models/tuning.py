"""Per-window XGBoost grid search.

Ported from ``notebooks/xgboost_parameter_tuning.ipynb`` into the production codebase.
For a given forecast window each candidate hyperparameter combination is scored by
holding out the window's own horizon (e.g. tune ``7d`` against a 168-hour holdout,
``1yr`` against a 365-day holdout), forecasting recursively, and averaging MAPE across
all regional series. The lowest-MAPE combination wins and is persisted as JSON so the
forecast job can reuse it without re-searching.

Grid search is expensive (combos x series x recursive horizon), so it is meant to run
periodically (e.g. a weekly Airflow DAG), not on every forecast.
"""

from __future__ import annotations

import itertools
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import storage
from src.models.xgboost_model import XgbConfig, _prepare_series, fit_and_forecast_series_with_info
from src.windows import ForecastWindow

LOGGER = logging.getLogger(__name__)

PARAM_KEYS = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
]

# Cartesian product searched per window (mirrors the tuning notebook's defaults).
DEFAULT_GRID: dict[str, list[float]] = {
    "n_estimators": [300, 600],
    "max_depth": [4, 6],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
    "min_child_weight": [1, 5],
}

_INT_PARAMS = {"n_estimators", "max_depth", "min_child_weight"}

# Cap the training history during tuning to keep the search tractable. Daily models
# need the full history (the 365-day lag), so 0 means "use everything".
_DEFAULT_MAX_TRAIN_STEPS = {"hourly": 1440, "daily": 0}


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100.0)


def _cast_params(params: dict[str, float]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in params.items():
        out[key] = int(value) if key in _INT_PARAMS else float(value)
    return out


def grid_search_window(
    load_df: pd.DataFrame,
    window: ForecastWindow,
    grid: dict[str, list[float]] | None = None,
    group_cols: tuple[str, ...] = ("region", "series"),
    max_train_steps: int | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Search ``grid`` for ``window`` and return ``(best_params, results_df)``.

    ``best_params`` is a plain dict of XGBoost hyperparameters; ``results_df`` holds the
    averaged MAPE/n_series for every evaluated combination, sorted ascending by MAPE.
    """
    grid = grid or DEFAULT_GRID
    group_cols = list(group_cols)
    horizon = window.horizon
    if max_train_steps is None:
        max_train_steps = _DEFAULT_MAX_TRAIN_STEPS.get(window.resolution, 0)

    df = load_df.copy()
    df["period"] = pd.to_datetime(df["period"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    base_config = XgbConfig.for_resolution(window.resolution)

    # Resample each series to the window resolution once; reuse across all combos.
    prepared: dict[tuple, pd.Series] = {}
    for group_values, gdf in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        ts = gdf.set_index("period")["value"]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index()
        clean = _prepare_series(ts, base_config)
        non_null = clean.dropna()
        if len(non_null) <= horizon + base_config.min_observations:
            continue
        prepared[group_values] = clean.interpolate(limit_direction="both").ffill().bfill()

    if not prepared:
        LOGGER.warning("No series with enough history to tune window %s", window.key)
        return _cast_params({k: grid[k][0] for k in PARAM_KEYS}), pd.DataFrame()

    combos = list(itertools.product(*[grid[k] for k in PARAM_KEYS]))
    LOGGER.info(
        "Grid search window=%s combos=%s series=%s horizon=%s (%s)",
        window.key,
        len(combos),
        len(prepared),
        horizon,
        window.resolution,
    )

    results: list[dict[str, float]] = []
    for combo in combos:
        params = dict(zip(PARAM_KEYS, combo))
        config = XgbConfig.for_resolution(window.resolution, **params)
        scores: list[float] = []
        for clean in prepared.values():
            non_null = clean.dropna()
            test_actual = non_null.iloc[-horizon:]
            train_end = test_actual.index[0] - config.step
            train_ts = clean.loc[:train_end]
            if max_train_steps and max_train_steps > 0:
                train_ts = train_ts.iloc[-max_train_steps:]
            preds, _ = fit_and_forecast_series_with_info(train_ts, horizon=horizon, config=config)
            y_pred = np.asarray(preds, dtype=float)[: len(test_actual)]
            y_true = test_actual.to_numpy(dtype=float)
            if len(y_pred) != len(y_true) or np.isnan(y_pred).any():
                continue
            scores.append(_mape(y_true, y_pred))
        if scores:
            results.append({**params, "mape": float(np.nanmean(scores)), "n_series": len(scores)})

    if not results:
        LOGGER.warning("Grid search produced no scored combos for window %s", window.key)
        return _cast_params({k: grid[k][0] for k in PARAM_KEYS}), pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("mape").reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_params = _cast_params({k: best_row[k] for k in PARAM_KEYS})
    LOGGER.info("Best params for window=%s: %s (mape=%.3f%%)", window.key, best_params, best_row["mape"])
    return best_params, results_df


def tune_and_save_window(
    load_df: pd.DataFrame,
    window: ForecastWindow,
    params_path: str | os.PathLike[str],
    grid: dict[str, list[float]] | None = None,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Run the grid search for ``window`` and persist the winner to ``params_path``."""
    best_params, results_df = grid_search_window(load_df, window, grid=grid)
    payload = {
        "window": window.key,
        "resolution": window.resolution,
        "horizon": window.horizon,
        "params": best_params,
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "best_mape": float(results_df.iloc[0]["mape"]) if not results_df.empty else None,
        "combos_evaluated": int(len(results_df)),
    }
    storage.write_json(params_path, payload)
    LOGGER.info("Saved best params for window=%s to %s", window.key, params_path)
    return best_params, results_df


def load_best_params(params_path: str | os.PathLike[str]) -> dict[str, float | int] | None:
    """Return persisted best params for a window, or None if not yet tuned."""
    if not storage.exists(params_path):
        return None
    payload = storage.read_json(params_path)
    params = payload.get("params") if isinstance(payload, dict) else None
    return params or None
