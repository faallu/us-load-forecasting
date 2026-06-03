import numpy as np
import pandas as pd

from src.models.evaluate import evaluate_load_backtest


def _hourly_history(periods: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="h", tz="UTC")


def test_evaluate_load_backtest_returns_rows() -> None:
    history = _hourly_history(200)
    load_df = pd.DataFrame(
        {
            "region": ["CISO"] * len(history),
            "series": ["CISO total"] * len(history),
            "period": history,
            "value": 100.0 + 10.0 * np.sin(np.arange(len(history)) * 2 * np.pi / 24.0),
        }
    )
    by_series, overall = evaluate_load_backtest(load_df, holdout_hours=24)
    assert len(by_series) == 1
    assert overall["n"].iloc[0] == 24
    assert "mae" in by_series.columns and "aic" in by_series.columns
