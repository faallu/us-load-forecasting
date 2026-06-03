from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd

from src import storage
from src.models.sarima import SarimaConfig, forecast_grouped_dataframe

if TYPE_CHECKING:
    from src.models.xgboost_model import XgbConfig


def build_load_forecast(
    load_df: pd.DataFrame,
    horizon_hours: int,
    seasonal_period: int = 24,
    max_train_hours: int | None = None,
    model: str = "sarima",
    xgb_config: "XgbConfig | None" = None,
) -> pd.DataFrame:
    df = load_df.copy()
    if max_train_hours is not None and max_train_hours > 0:
        df["period"] = pd.to_datetime(df["period"], utc=True)
        cutoff = df["period"].max() - pd.Timedelta(hours=max_train_hours - 1)
        df = df.loc[df["period"] >= cutoff].copy()

    if model == "xgboost":
        # Imported lazily so SARIMA-only deployments do not require the xgboost package.
        from src.models.xgboost_model import XgbConfig, forecast_grouped_dataframe as xgb_forecast_grouped

        return xgb_forecast_grouped(
            df=df,
            group_cols=["region", "series"],
            value_col="value",
            horizon=horizon_hours,
            issue_time=None,
            config=xgb_config or XgbConfig(),
        )

    config = SarimaConfig(order=(1, 1, 2), seasonal_order=(1, 1, 1, seasonal_period))
    return forecast_grouped_dataframe(
        df=df,
        group_cols=["region", "series"],
        value_col="value",
        horizon=horizon_hours,
        issue_time=None,
        config=config,
    )


def save_forecast(df: pd.DataFrame, output_path: str | os.PathLike[str]) -> None:
    storage.write_parquet(df, output_path)
