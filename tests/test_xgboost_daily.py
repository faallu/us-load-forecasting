import numpy as np
import pandas as pd

from src.models.predict import build_load_forecast
from src.models.xgboost_model import (
    DAILY_CALENDAR_COLS,
    XgbConfig,
)


def _daily_history(days: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01T00:00:00Z", periods=days, freq="D", tz="UTC")


def _daily_values(n: int) -> np.ndarray:
    idx = np.arange(n)
    weekly = 200000 + 30000 * np.sin(2 * np.pi * idx / 7)
    annual = 50000 * np.sin(2 * np.pi * idx / 365)
    return weekly + annual


def test_for_resolution_daily_feature_cols() -> None:
    config = XgbConfig.for_resolution("daily", n_estimators=50, max_depth=3)
    assert config.resolution == "daily"
    assert config.freq == "D"
    assert config.lags == (1, 2, 3, 7, 14, 28, 365)
    for col in DAILY_CALENDAR_COLS:
        assert col in config.feature_cols
    assert config.n_estimators == 50 and config.max_depth == 3


def test_daily_window_forecast_shape_and_spacing() -> None:
    history = _daily_history(500)
    load_df = pd.DataFrame(
        {
            "region": ["CISO"] * len(history),
            "series": ["CISO total"] * len(history),
            "period": history,
            "value": _daily_values(len(history)),
        }
    )
    config = XgbConfig.for_resolution("daily", n_estimators=40, max_depth=3)
    out = build_load_forecast(load_df, horizon_hours=30, model="xgboost", xgb_config=config)

    assert len(out) == 30
    assert {"region", "series", "issue_time", "target_time", "yhat"} == set(out.columns)
    assert out["yhat"].notna().all()
    # Daily resolution => target times are one day apart.
    target_times = pd.to_datetime(out["target_time"], utc=True).sort_values().reset_index(drop=True)
    deltas = target_times.diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(days=1)
