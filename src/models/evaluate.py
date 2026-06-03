from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src import storage
from src.models.sarima import SarimaConfig, fit_and_forecast_series_with_info


def _smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    valid = denom != 0
    if not valid.any():
        return 0.0
    return float((np.abs(y_true[valid] - y_pred[valid]) / denom[valid]).mean() * 100.0)


def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = y_true != 0
    if not valid.any():
        return 0.0
    return float((np.abs((y_true[valid] - y_pred[valid]) / y_true[valid]).mean()) * 100.0)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
    }


def _merge_actual_and_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    group_cols: list[str],
    actual_time_col: str,
    actual_value_col: str,
    forecast_time_col: str,
    forecast_value_col: str,
) -> pd.DataFrame:
    actual = actual_df.copy()
    fcst = forecast_df.copy()

    actual[actual_time_col] = pd.to_datetime(actual[actual_time_col], utc=True)
    fcst[forecast_time_col] = pd.to_datetime(fcst[forecast_time_col], utc=True)
    actual[actual_value_col] = pd.to_numeric(actual[actual_value_col], errors="coerce")
    fcst[forecast_value_col] = pd.to_numeric(fcst[forecast_value_col], errors="coerce")

    merged = actual.merge(
        fcst,
        how="inner",
        left_on=[*group_cols, actual_time_col],
        right_on=[*group_cols, forecast_time_col],
        suffixes=("_actual", "_forecast"),
    )
    merged = merged.dropna(subset=[actual_value_col, forecast_value_col])
    return merged


def evaluate_forecast(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    group_cols: list[str],
    actual_time_col: str = "period",
    actual_value_col: str = "value",
    forecast_time_col: str = "target_time",
    forecast_value_col: str = "yhat",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-group and overall metrics by joining actual period with forecast target_time."""
    merged = _merge_actual_and_forecast(
        actual_df=actual_df,
        forecast_df=forecast_df,
        group_cols=group_cols,
        actual_time_col=actual_time_col,
        actual_value_col=actual_value_col,
        forecast_time_col=forecast_time_col,
        forecast_value_col=forecast_value_col,
    )

    if merged.empty:
        empty = pd.DataFrame(columns=[*group_cols, "mae", "rmse", "mape", "smape", "n"])
        overall = pd.DataFrame(columns=["mae", "rmse", "mape", "smape", "n"])
        return empty, overall

    rows: list[dict[str, object]] = []
    for key_vals, gdf in merged.groupby(group_cols):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        y_true = gdf[actual_value_col].astype(float)
        y_pred = gdf[forecast_value_col].astype(float)
        score = _metrics(y_true, y_pred)
        row = {col: key_vals[idx] for idx, col in enumerate(group_cols)}
        row.update(score)
        row["n"] = int(len(gdf))
        rows.append(row)

    by_series = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    overall_score = _metrics(
        merged[actual_value_col].astype(float),
        merged[forecast_value_col].astype(float),
    )
    overall = pd.DataFrame([{**overall_score, "n": int(len(merged))}])
    return by_series, overall


def evaluate_load_backtest(
    load_df: pd.DataFrame,
    holdout_hours: int,
    group_cols: tuple[str, str] = ("region", "series"),
    value_col: str = "value",
    period_col: str = "period",
    config: SarimaConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``holdout_hours`` per series, refit SARIMAX on prior history, score point forecasts.

    Saved operational forecasts start at ``last_actual + 1h``, so they do not overlap processed
    actuals until new hours are ingested; this backtest gives MAE / RMSE / MAPE on recent history.
    """
    config = config or SarimaConfig()
    if holdout_hours < 1:
        raise ValueError("holdout_hours must be >= 1")

    local = load_df.copy()
    local[period_col] = pd.to_datetime(local[period_col], utc=True)
    list_cols = list(group_cols)

    rows: list[dict[str, object]] = []
    pooled_true: list[float] = []
    pooled_pred: list[float] = []
    for group_values, group_df in local.groupby(list_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        ts = group_df.set_index(period_col)[value_col]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index().asfreq("h")
        non_null_ts = ts.dropna()
        if non_null_ts.empty:
            continue
        if len(non_null_ts) < holdout_hours + config.min_observations:
            continue

        first_holdout_time = non_null_ts.index[-holdout_hours]
        train_end = first_holdout_time - pd.Timedelta(hours=1)
        train_ts = ts.loc[:train_end]
        y_true = non_null_ts.iloc[-holdout_hours:].astype(float)
        preds, info = fit_and_forecast_series_with_info(train_ts, horizon=holdout_hours, config=config)
        y_pred = preds.astype(float).reset_index(drop=True)
        y_true_arr = y_true.to_numpy()
        if len(y_pred) != len(y_true_arr):
            continue

        y_t = pd.Series(y_true_arr)
        y_p = pd.Series(y_pred)
        score = _metrics(y_t, y_p)
        row = {col: group_values[idx] for idx, col in enumerate(list_cols)}
        row.update(score)
        row["n"] = int(len(y_true_arr))
        if info:
            row["aic"] = info.get("aic")
            row["bic"] = info.get("bic")
            row["hqic"] = info.get("hqic")
        else:
            row["aic"] = float("nan")
            row["bic"] = float("nan")
            row["hqic"] = float("nan")
        rows.append(row)
        pooled_true.extend(float(x) for x in y_true_arr)
        pooled_pred.extend(float(x) for x in y_pred.to_numpy())

    if not rows:
        cols = [
            *list_cols,
            "mae",
            "rmse",
            "mape",
            "smape",
            "n",
            "aic",
            "bic",
            "hqic",
        ]
        empty = pd.DataFrame(columns=cols)
        overall = pd.DataFrame(columns=["mae", "rmse", "mape", "smape", "n", "aic", "bic", "hqic"])
        return empty, overall

    by_series = pd.DataFrame(rows).sort_values(list_cols).reset_index(drop=True)
    overall_score = _metrics(pd.Series(pooled_true), pd.Series(pooled_pred))
    overall = pd.DataFrame(
        [
            {
                **overall_score,
                "n": len(pooled_true),
                "aic_mean_by_series": float(np.nanmean(by_series["aic"].to_numpy(dtype=float))),
                "bic_mean_by_series": float(np.nanmean(by_series["bic"].to_numpy(dtype=float))),
            }
        ]
    )
    return by_series, overall


def plot_forecast_vs_actual(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    group_cols: list[str],
    output_path: str | os.PathLike[str],
    actual_time_col: str = "period",
    actual_value_col: str = "value",
    forecast_time_col: str = "target_time",
    forecast_value_col: str = "yhat",
    max_groups: int = 6,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    merged = _merge_actual_and_forecast(
        actual_df=actual_df,
        forecast_df=forecast_df,
        group_cols=group_cols,
        actual_time_col=actual_time_col,
        actual_value_col=actual_value_col,
        forecast_time_col=forecast_time_col,
        forecast_value_col=forecast_value_col,
    )

    if merged.empty:
        raise ValueError("No overlapping actual and forecast rows to plot.")

    counts = (
        merged.groupby(group_cols, dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(max_groups)
        .reset_index(name="n")
    )

    selected_keys = counts[group_cols].to_dict("records")
    n_plots = len(selected_keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, max(4, n_plots * 3)), sharex=False)
    if n_plots == 1:
        axes = [axes]

    for idx, keys in enumerate(selected_keys):
        mask = pd.Series(True, index=merged.index)
        for col, val in keys.items():
            mask &= merged[col] == val
        sdf = merged.loc[mask].sort_values(actual_time_col)

        ax = axes[idx]
        ax.plot(sdf[actual_time_col], sdf[actual_value_col], label="actual", linewidth=2)
        ax.plot(sdf[forecast_time_col], sdf[forecast_value_col], label="forecast", linewidth=2, linestyle="--")
        title = ", ".join(f"{k}={v}" for k, v in keys.items())
        ax.set_title(title)
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (UTC)")
    fig.tight_layout()

    saved = storage.save_figure(fig, output_path, dpi=150)
    plt.close(fig)
    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate forecast metrics against actuals.")
    parser.add_argument("--actual", required=True, help="Path to actuals parquet file.")
    parser.add_argument("--forecast", required=True, help="Path to forecast parquet file.")
    parser.add_argument(
        "--kind",
        choices=["load"],
        required=True,
        help="Load uses group keys [region, series].",
    )
    parser.add_argument("--output", default="", help="Optional path to write per-series metrics parquet.")
    parser.add_argument("--plot-output", default="", help="Optional image path to save actual-vs-forecast plots.")
    parser.add_argument(
        "--plot-max-series",
        type=int,
        default=6,
        help="Max number of top groups to include in the comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    actual = pd.read_parquet(args.actual)
    forecast = pd.read_parquet(args.forecast)

    group_cols = ["region", "series"]
    by_series, overall = evaluate_forecast(actual, forecast, group_cols=group_cols)

    print("Overall metrics:")
    print(overall.to_string(index=False))
    print("")
    print("Per-series metrics (first 20):")
    print(by_series.head(20).to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        by_series.to_parquet(output_path, index=False)
        print("")
        print(f"Saved per-series metrics to {output_path}")

    if args.plot_output:
        plot_path = Path(args.plot_output)
        saved = plot_forecast_vs_actual(
            actual_df=actual,
            forecast_df=forecast,
            group_cols=group_cols,
            output_path=plot_path,
            max_groups=max(1, args.plot_max_series),
        )
        print("")
        print(f"Saved forecast comparison plot to {saved}")


if __name__ == "__main__":
    main()
