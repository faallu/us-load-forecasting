import pandas as pd

from src.models.predict import build_load_forecast


def _hourly_history(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="h", tz="UTC")


def test_load_forecast_shape() -> None:
    history = _hourly_history(120)
    load_df = pd.DataFrame(
        {
            "region": ["CISO"] * len(history),
            "series": ["CISO total"] * len(history),
            "period": history,
            "value": [100 + (i % 24) for i in range(len(history))],
        }
    )
    out = build_load_forecast(load_df, horizon_hours=24)
    assert len(out) == 24
    assert {"region", "series", "issue_time", "target_time", "yhat"} == set(out.columns)
