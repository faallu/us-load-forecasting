from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src import storage
from src.models.evaluate import evaluate_forecast, plot_forecast_vs_actual
from src.models.sarima import SarimaConfig, fit_and_forecast_series_with_info

if TYPE_CHECKING:
    from src.models.xgboost_model import XgbConfig

_SARIMA_INFO_COLS = ["aic", "bic", "hqic", "llf"]
_XGB_INFO_COLS = ["train_rows", "n_features"]


def _fit_forecast(
    train_ts: pd.Series,
    horizon: int,
    model: str,
    sarima_config: SarimaConfig,
    xgb_config: "XgbConfig | None",
) -> tuple[pd.Series, dict[str, object]]:
    """Dispatch to the selected model and keep only that model's info columns."""
    if model == "xgboost":
        from src.models.xgboost_model import XgbConfig
        from src.models.xgboost_model import fit_and_forecast_series_with_info as xgb_fit

        preds, info = xgb_fit(train_ts, horizon=horizon, config=xgb_config or XgbConfig())
        return preds, {col: info.get(col) for col in _XGB_INFO_COLS}

    preds, info = fit_and_forecast_series_with_info(train_ts, horizon=horizon, config=sarima_config)
    return preds, {col: info.get(col) for col in _SARIMA_INFO_COLS}


def build_holdout_backtest_forecast(
    actual_df: pd.DataFrame,
    group_cols: list[str],
    holdout_hours: int,
    max_train_hours: int | None = None,
    value_col: str = "value",
    seasonal_period: int = 24,
    model: str = "sarima",
    xgb_config: "XgbConfig | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_hours < 1:
        raise ValueError("holdout_hours must be >= 1")
    if max_train_hours is not None and max_train_hours < 1:
        raise ValueError("max_train_hours must be >= 1 when provided")

    sarima_config = SarimaConfig(order=(1, 1, 2), seasonal_order=(1, 1, 1, seasonal_period))
    info_value_cols = _XGB_INFO_COLS if model == "xgboost" else _SARIMA_INFO_COLS
    local = actual_df.copy()
    local["period"] = pd.to_datetime(local["period"], utc=True)
    local[value_col] = pd.to_numeric(local[value_col], errors="coerce")

    forecast_rows: list[dict[str, object]] = []
    info_rows: list[dict[str, object]] = []

    for group_values, gdf in local.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        ts = gdf.set_index("period")[value_col]
        ts = ts[~ts.index.duplicated(keep="last")].sort_index().asfreq("h")
        non_null = ts.dropna()
        if non_null.empty:
            continue
        if len(non_null) <= holdout_hours:
            continue

        test_start = non_null.index.max() - pd.Timedelta(hours=holdout_hours - 1)
        train_end = test_start - pd.Timedelta(hours=1)
        train_ts = ts.loc[:train_end]
        if max_train_hours is not None:
            train_start = train_end - pd.Timedelta(hours=max_train_hours - 1)
            train_ts = train_ts.loc[train_start:train_end]
        train_non_null = train_ts.dropna()
        if train_non_null.empty:
            continue

        preds, info = _fit_forecast(train_ts, holdout_hours, model, sarima_config, xgb_config)
        issue_time = train_non_null.index.max()
        start_time = issue_time + pd.Timedelta(hours=1)

        base_row = {col: group_values[idx] for idx, col in enumerate(group_cols)}
        info_rows.append(
            {
                **base_row,
                "issue_time": issue_time,
                "holdout_hours": holdout_hours,
                "train_points": int(len(train_non_null)),
                **{col: info.get(col) for col in info_value_cols},
            }
        )

        for step, yhat in enumerate(preds):
            forecast_rows.append(
                {
                    **base_row,
                    "issue_time": issue_time,
                    "target_time": start_time + pd.Timedelta(hours=step),
                    "yhat": float(yhat),
                }
            )

    forecast_df = pd.DataFrame(forecast_rows)
    if forecast_df.empty:
        forecast_df = pd.DataFrame(columns=[*group_cols, "issue_time", "target_time", "yhat"])
    else:
        forecast_df = forecast_df.sort_values([*group_cols, "target_time"]).reset_index(drop=True)

    info_df = pd.DataFrame(info_rows)
    if info_df.empty:
        info_df = pd.DataFrame(
            columns=[*group_cols, "issue_time", "holdout_hours", "train_points", *info_value_cols]
        )
    else:
        info_df = info_df.sort_values(group_cols).reset_index(drop=True)

    return forecast_df, info_df


def _build_text_report(
    kind: str,
    holdout_hours: int,
    by_series: pd.DataFrame,
    overall: pd.DataFrame,
    output_path: str | os.PathLike[str],
    header_note: str | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"Backtest metrics ({kind})")
    lines.append(header_note if header_note is not None else f"Holdout hours: {holdout_hours}")
    lines.append("")
    lines.append("Overall metrics")
    if overall.empty:
        lines.append("No overlapping rows for metric calculation.")
    else:
        lines.append(overall.to_string(index=False))
    lines.append("")
    lines.append("Per-series metrics")
    if by_series.empty:
        lines.append("No per-series results.")
    else:
        preferred_cols = [col for col in ["region", "series", "fuel", "n", "mae", "rmse", "mape", "smape", "aic", "bic", "hqic"] if col in by_series.columns]
        lines.append(by_series[preferred_cols].to_string(index=False))
    storage.write_text(output_path, "\n".join(lines))


def run_backtest(
    actual_path: str | os.PathLike[str],
    kind: str,
    holdout_hours: int,
    metrics_txt_path: str | os.PathLike[str],
    max_train_hours: int | None = None,
    forecast_output: str | os.PathLike[str] | None = None,
    metrics_output: str | os.PathLike[str] | None = None,
    plot_output: str | os.PathLike[str] | None = None,
    plot_max_series: int = 6,
    model: str = "sarima",
    xgb_config: "XgbConfig | None" = None,
    test_fraction: float = 0.0,
    resolution: str = "hourly",
    window_label: str | None = None,
) -> dict[str, str]:
    actual = storage.read_parquet(actual_path)
    group_cols = ["region", "series"] if kind == "load" else ["region", "fuel"]

    header_note: str | None = None
    if model == "xgboost" and test_fraction > 0:
        # Chronological train/test split (true lagged features) instead of a recursive holdout.
        # Daily windows read the daily demand series directly, so the actuals already match
        # the forecast resolution and no aggregation is needed here.
        from src.models.xgboost_model import XgbConfig, build_split_forecast

        config = xgb_config or XgbConfig()
        backtest_forecast, info_df = build_split_forecast(
            actual_df=actual,
            group_cols=group_cols,
            test_fraction=test_fraction,
            config=config,
        )
        train_pct = round((1.0 - test_fraction) * 100)
        test_pct = round(test_fraction * 100)
        window_prefix = f"Window: {window_label} ({config.resolution}). " if window_label else ""
        header_note = (
            f"{window_prefix}Train/test split: {train_pct}% train / {test_pct}% test "
            "(chronological, true lagged features)"
        )
    else:
        backtest_forecast, info_df = build_holdout_backtest_forecast(
            actual_df=actual,
            group_cols=group_cols,
            holdout_hours=holdout_hours,
            max_train_hours=max_train_hours,
            model=model,
            xgb_config=xgb_config,
        )

    by_series, overall = evaluate_forecast(
        actual_df=actual,
        forecast_df=backtest_forecast,
        group_cols=group_cols,
    )
    by_series = by_series.merge(info_df, on=group_cols, how="left")

    _build_text_report(
        kind=kind,
        holdout_hours=holdout_hours,
        by_series=by_series,
        overall=overall,
        output_path=metrics_txt_path,
        header_note=header_note,
    )

    saved: dict[str, str] = {"metrics_txt": str(metrics_txt_path)}

    if forecast_output is not None:
        storage.write_parquet(backtest_forecast, forecast_output)
        saved["forecast"] = str(forecast_output)

    if metrics_output is not None:
        storage.write_parquet(by_series, metrics_output)
        saved["metrics"] = str(metrics_output)

    if plot_output is not None:
        saved_plot = plot_forecast_vs_actual(
            actual_df=actual,
            forecast_df=backtest_forecast,
            group_cols=group_cols,
            output_path=plot_output,
            max_groups=max(1, plot_max_series),
        )
        saved["plot"] = str(saved_plot)

    return saved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run holdout backtest and save metric reports.")
    parser.add_argument("--actual", required=True, help="Path to actuals parquet file.")
    parser.add_argument("--kind", choices=["load"], required=True)
    parser.add_argument("--holdout-hours", type=int, default=168, help="Number of recent hours to backtest.")
    parser.add_argument(
        "--max-train-hours",
        type=int,
        default=0,
        help="Optional cap on training history hours (0 uses full history).",
    )
    parser.add_argument("--metrics-txt", required=True, help="Output path for plain-text metrics report.")
    parser.add_argument("--forecast-output", default="", help="Optional path to save backtest forecast parquet.")
    parser.add_argument("--metrics-output", default="", help="Optional path to save per-series backtest metrics parquet.")
    parser.add_argument("--plot-output", default="", help="Optional path to save backtest plot image.")
    parser.add_argument("--plot-max-series", type=int, default=6, help="Max number of groups to include in the plot.")
    parser.add_argument(
        "--model",
        choices=["sarima", "xgboost"],
        default="sarima",
        help="Forecasting model to backtest.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved = run_backtest(
        actual_path=Path(args.actual),
        kind=args.kind,
        holdout_hours=args.holdout_hours,
        metrics_txt_path=Path(args.metrics_txt),
        max_train_hours=args.max_train_hours if args.max_train_hours > 0 else None,
        forecast_output=Path(args.forecast_output) if args.forecast_output else None,
        metrics_output=Path(args.metrics_output) if args.metrics_output else None,
        plot_output=Path(args.plot_output) if args.plot_output else None,
        plot_max_series=args.plot_max_series,
        model=args.model,
    )

    print("Saved outputs:")
    for key, path in saved.items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
