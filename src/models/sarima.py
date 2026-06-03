from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SarimaConfig:
    order: tuple[int, int, int] = (1, 1, 2)
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24)
    min_observations: int = 96


def fit_and_forecast_series(
    series: pd.Series,
    horizon: int,
    config: SarimaConfig | None = None,
) -> pd.Series:
    forecast, _ = fit_and_forecast_series_with_info(series, horizon, config=config)
    return forecast


def fit_and_forecast_series_with_info(
    series: pd.Series,
    horizon: int,
    config: SarimaConfig | None = None,
) -> tuple[pd.Series, dict[str, float | None]]:
    """Point forecast plus in-sample information criteria from the fitted SARIMAX (when used)."""
    config = config or SarimaConfig()
    clean = series.sort_index().astype(float)
    clean = clean.asfreq("h")
    if clean.isna().all():
        return pd.Series([0.0] * horizon), {}
    # Keep hourly frequency to avoid statsmodels date-index warnings.
    clean = clean.interpolate(limit_direction="both").ffill().bfill()
    if len(clean) < config.min_observations:
        # Fallback for short series: last observed value persistence.
        if clean.empty:
            return pd.Series([0.0] * horizon), {}
        last = float(clean.iloc[-1])
        return pd.Series([last] * horizon), {}

    model = SARIMAX(
        clean,
        order=config.order,
        seasonal_order=config.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(steps=horizon)
    info = {
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "hqic": float(fit.hqic),
        "llf": float(fit.llf),
    }
    return pd.Series(forecast).astype(float).reset_index(drop=True), info


def forecast_grouped_dataframe(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    horizon: int,
    issue_time: pd.Timestamp | None = None,
    config: SarimaConfig | None = None,
) -> pd.DataFrame:
    if df.empty:
        cols = [*group_cols, "issue_time", "target_time", "yhat"]
        return pd.DataFrame(columns=cols)

    out_rows: list[dict[str, object]] = []
    local = df.copy()
    local["period"] = pd.to_datetime(local["period"], utc=True)

    for group_values, group_df in local.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        ts = group_df.set_index("period")[value_col]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index().asfreq("h")
        non_null_ts = ts.dropna()
        if non_null_ts.empty:
            continue
        preds, _ = fit_and_forecast_series_with_info(ts, horizon=horizon, config=config)
        forecast_origin = non_null_ts.index.max()
        start_time = forecast_origin + pd.Timedelta(hours=1)

        for step, yhat in enumerate(preds):
            row = {col: group_values[idx] for idx, col in enumerate(group_cols)}
            row["issue_time"] = forecast_origin if issue_time is None else issue_time
            row["target_time"] = start_time + pd.Timedelta(hours=step)
            row["yhat"] = float(yhat)
            out_rows.append(row)

    return pd.DataFrame(out_rows).sort_values([*group_cols, "target_time"]).reset_index(drop=True)
