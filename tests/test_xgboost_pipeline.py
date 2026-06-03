import numpy as np
import pandas as pd

from src.models.predict import build_load_forecast
from src.models.xgboost_model import XgbConfig, fit_and_forecast_series_with_info


def _hourly_history(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="h", tz="UTC")


def _seasonal_series(periods: int) -> list[float]:
    idx = np.arange(periods)
    daily = 100 + 20 * np.sin(2 * np.pi * idx / 24)
    weekly = 5 * np.sin(2 * np.pi * idx / 168)
    return list(daily + weekly)


def test_xgboost_load_forecast_shape() -> None:
    history = _hourly_history(400)
    load_df = pd.DataFrame(
        {
            "region": ["CISO"] * len(history),
            "series": ["CISO total"] * len(history),
            "period": history,
            "value": _seasonal_series(len(history)),
        }
    )
    out = build_load_forecast(load_df, horizon_hours=24, model="xgboost")
    assert len(out) == 24
    assert {"region", "series", "issue_time", "target_time", "yhat"} == set(out.columns)
    assert out["yhat"].notna().all()


def test_xgboost_persistence_fallback_for_short_series() -> None:
    history = _hourly_history(10)
    series = pd.Series([50.0] * len(history), index=history)
    preds, info = fit_and_forecast_series_with_info(series, horizon=6, config=XgbConfig())
    assert len(preds) == 6
    # Too short to train: falls back to last-value persistence and reports no training info.
    assert info == {}
    assert all(p == 50.0 for p in preds)
