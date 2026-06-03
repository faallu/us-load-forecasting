"""FastAPI service that serves precomputed XGBoost forecasts from storage (S3).

This is a *serve-only* layer: it never trains and deliberately avoids importing the
modelling code (xgboost/statsmodels), keeping the Lambda image small. All heavy work
is done by Airflow, which writes per-window forecasts and metrics to S3; this app
reads, filters, and returns them as JSON.

Deployed to AWS Lambda via the ``handler`` (Mangum) below; also runnable locally with
``uvicorn api.main:app --reload``.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import storage
from src.config import get_settings
from src.windows import FORECAST_WINDOWS, get_window, window_keys

app = FastAPI(
    title="Load Forecasting API",
    description="Serves precomputed EIA regional load forecasts (XGBoost) by window and region.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Small in-process cache so warm Lambda invocations don't re-download parquet on
# every request. Keyed by storage path; entries expire after _CACHE_TTL seconds.
_CACHE_TTL = 300.0
_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _read_parquet_cached(path: str) -> pd.DataFrame:
    now = time.time()
    cached = _cache.get(path)
    if cached is not None and now - cached[0] < _CACHE_TTL:
        return cached[1]
    if not storage.exists(path):
        raise FileNotFoundError(path)
    df = storage.read_parquet(path)
    _cache[path] = (now, df)
    return df


def _records(df: pd.DataFrame) -> list[dict]:
    safe = df.replace({np.nan: None})
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].apply(lambda v: v.isoformat() if pd.notna(v) else None)
    return safe.to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "storage_backend": settings.storage_backend, "data_root": settings.data_root}


@app.get("/windows")
def list_windows() -> dict:
    return {
        "windows": [
            {
                "key": w.key,
                "label": w.label,
                "horizon": w.horizon,
                "resolution": w.resolution,
                "horizon_hours": w.horizon_hours,
            }
            for w in FORECAST_WINDOWS.values()
        ]
    }


@app.get("/regions")
def list_regions() -> dict:
    return {"regions": list(get_settings().allowed_regions)}


@app.get("/forecast")
def get_forecast(
    window: str = Query(..., description=f"One of: {', '.join(window_keys())}"),
    region: str | None = Query(None, description="Optional region filter, e.g. CISO"),
) -> dict:
    try:
        win = get_window(window)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    path = settings.forecast_path(win.key)
    try:
        df = _read_parquet_cached(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No forecast available for window '{win.key}' yet.") from exc

    if region is not None and not df.empty:
        df = df.loc[df["region"].str.upper() == region.upper()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No forecast rows for region '{region}'.")

    issue_time = None
    if not df.empty and "issue_time" in df.columns:
        latest = pd.to_datetime(df["issue_time"], utc=True).max()
        df = df.loc[pd.to_datetime(df["issue_time"], utc=True) == latest]
        issue_time = latest.isoformat()

    return {
        "window": win.key,
        "resolution": win.resolution,
        "region": region,
        "issue_time": issue_time,
        "count": int(len(df)),
        "forecast": _records(df),
    }


@app.get("/metrics")
def get_metrics(
    window: str = Query(..., description=f"One of: {', '.join(window_keys())}"),
) -> dict:
    try:
        win = get_window(window)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    parquet_path = settings.backtest_metrics_parquet_path(win.key)
    text_path = settings.backtest_metrics_path(win.key)

    by_series: list[dict] = []
    try:
        by_series = _records(_read_parquet_cached(parquet_path))
    except FileNotFoundError:
        by_series = []

    report = None
    if storage.exists(text_path):
        report = storage.read_text(text_path)

    if not by_series and report is None:
        raise HTTPException(status_code=404, detail=f"No metrics available for window '{win.key}' yet.")

    return {"window": win.key, "resolution": win.resolution, "by_series": by_series, "report": report}


# AWS Lambda entry point (API Gateway HTTP API or Lambda Function URL).
try:
    from mangum import Mangum

    handler = Mangum(app)
except ModuleNotFoundError:  # pragma: no cover - mangum only required in Lambda
    handler = None
