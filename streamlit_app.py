from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ACTUAL_PATH = ROOT / "data" / "processed" / "load_hourly.parquet"
FORECAST_PATH = ROOT / "data" / "forecasts" / "load_forecast.parquet"
BACKTEST_FORECAST_PATH = ROOT / "data" / "forecasts" / "load_backtest_forecast.parquet"
METRICS_PATH = ROOT / "data" / "metrics" / "load_backtest_metrics_168h.txt"


st.set_page_config(
    page_title="Load Forecast Operations",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_backtest_metrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), ""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    overall = _parse_overall_metrics(lines)
    by_series = _parse_per_series_metrics(lines)
    return overall, by_series, text


def _parse_overall_metrics(lines: list[str]) -> pd.DataFrame:
    for idx, line in enumerate(lines):
        if line.strip() == "Overall metrics" and idx + 2 < len(lines):
            headers = lines[idx + 1].split()
            values = lines[idx + 2].split()
            if len(headers) == len(values):
                return _coerce_numeric(pd.DataFrame([dict(zip(headers, values, strict=True))]))
    return pd.DataFrame(columns=["mae", "rmse", "mape", "smape", "n"])


def _parse_per_series_metrics(lines: list[str]) -> pd.DataFrame:
    pattern = re.compile(
        r"^\s*(?P<region>\S+)\s+"
        r"(?P<series>.+?)\s+"
        r"(?P<n>\d+)\s+"
        r"(?P<mae>[-+]?\d*\.?\d+)\s+"
        r"(?P<rmse>[-+]?\d*\.?\d+)\s+"
        r"(?P<mape>[-+]?\d*\.?\d+)\s+"
        r"(?P<smape>[-+]?\d*\.?\d+)\s+"
        r"(?P<aic>[-+]?\d*\.?\d+)\s+"
        r"(?P<bic>[-+]?\d*\.?\d+)\s+"
        r"(?P<hqic>[-+]?\d*\.?\d+)\s*$"
    )
    rows: list[dict[str, str]] = []
    in_section = False
    for line in lines:
        if line.strip() == "Per-series metrics":
            in_section = True
            continue
        if not in_section or not line.strip() or line.lstrip().startswith("region"):
            continue
        match = pattern.match(line)
        if match:
            rows.append(match.groupdict())

    if not rows:
        return pd.DataFrame(columns=["region", "series", "n", "mae", "rmse", "mape", "smape", "aic", "bic", "hqic"])
    return _coerce_numeric(pd.DataFrame(rows))


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["n", "mae", "rmse", "mape", "smape", "aic", "bic", "hqic"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_actuals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    local = df.copy()
    local["period"] = pd.to_datetime(local["period"], utc=True)
    local["value"] = pd.to_numeric(local["value"], errors="coerce")
    return local.dropna(subset=["period", "value"])


def prepare_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    local = df.copy()
    local["issue_time"] = pd.to_datetime(local["issue_time"], utc=True)
    local["target_time"] = pd.to_datetime(local["target_time"], utc=True)
    local["yhat"] = pd.to_numeric(local["yhat"], errors="coerce")
    return local.dropna(subset=["issue_time", "target_time", "yhat"])


def latest_forecast_run(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    latest_issue = df["issue_time"].max()
    return df.loc[df["issue_time"] == latest_issue].copy()


def format_time(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def format_number(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):,.{decimals}f}"


def format_percent(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):,.{decimals}f}%"


def freshness_status(latest_actual: pd.Timestamp | None) -> tuple[str, str]:
    if latest_actual is None or pd.isna(latest_actual):
        return "Missing", "No actual load data is available."

    now = pd.Timestamp.now(tz="UTC")
    age_hours = (now - latest_actual).total_seconds() / 3600
    if age_hours <= 3:
        return "Fresh", f"Latest actual is {age_hours:.1f} hours old."
    if age_hours <= 30:
        return "Delayed", f"Latest actual is {age_hours:.1f} hours old."
    return "Stale", f"Latest actual is {age_hours:.1f} hours old."


def show_status_message(status: str, message: str) -> None:
    if status == "Fresh":
        st.success(message)
    elif status == "Delayed":
        st.warning(message)
    else:
        st.error(message)


def build_backtest_overlay_chart(
    actuals: pd.DataFrame,
    backtest_forecasts: pd.DataFrame,
    region: str,
    pre_holdout_days: int = 7,
) -> pd.DataFrame:
    """Build a chart DataFrame that overlays backtest predictions on actuals for the holdout window.

    Shows `pre_holdout_days` of history before the holdout period so the model's
    entry point is visible.
    """
    region_bt = backtest_forecasts.loc[backtest_forecasts["region"] == region].copy()
    if region_bt.empty:
        return pd.DataFrame()

    holdout_start = region_bt["target_time"].min()
    history_cutoff = holdout_start - pd.Timedelta(days=pre_holdout_days)

    actual_region = actuals.loc[
        (actuals["region"] == region) & (actuals["period"] >= history_cutoff)
    ].copy()

    actual_chart = actual_region.rename(columns={"period": "time", "value": "mw"})[["time", "mw"]]
    actual_chart["type"] = "actual"

    backtest_chart = region_bt.rename(columns={"target_time": "time", "yhat": "mw"})[["time", "mw"]]
    backtest_chart["type"] = "backtest forecast"

    return pd.concat([actual_chart, backtest_chart], ignore_index=True).sort_values("time")


def build_actual_forecast_chart(actuals: pd.DataFrame, forecasts: pd.DataFrame, region: str, history_days: int) -> pd.DataFrame:
    actual_region = actuals.loc[actuals["region"] == region].copy()
    forecast_region = forecasts.loc[forecasts["region"] == region].copy()

    if not actual_region.empty:
        cutoff = actual_region["period"].max() - pd.Timedelta(days=history_days)
        actual_region = actual_region.loc[actual_region["period"] >= cutoff]

    actual_chart = actual_region.rename(columns={"period": "time", "value": "mw"})[["time", "mw"]]
    actual_chart["type"] = "actual"

    forecast_chart = forecast_region.rename(columns={"target_time": "time", "yhat": "mw"})[["time", "mw"]]
    forecast_chart["type"] = "forecast"

    return pd.concat([actual_chart, forecast_chart], ignore_index=True).sort_values("time")


def show_missing_data_help(actuals: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    if not actuals.empty and not forecasts.empty:
        return

    st.info(
        "Dashboard data is missing. Run `python -m src.jobs.run_ingest` and "
        "`python -m src.jobs.run_train_and_forecast` before deploying, or let GitHub Actions generate these outputs."
    )


actuals = prepare_actuals(load_parquet(ACTUAL_PATH))
forecasts_all = prepare_forecasts(load_parquet(FORECAST_PATH))
forecasts = latest_forecast_run(forecasts_all)
backtest_forecasts = prepare_forecasts(load_parquet(BACKTEST_FORECAST_PATH))
overall_metrics, series_metrics, metrics_text = load_backtest_metrics(METRICS_PATH)

st.title("Load Forecast Operations")
st.caption("Operational dashboard for EIA regional load actuals, SARIMA forecasts, and recent backtest performance.")

show_missing_data_help(actuals, forecasts)

regions = sorted(set(actuals.get("region", pd.Series(dtype=str))).union(set(forecasts.get("region", pd.Series(dtype=str)))))

with st.sidebar:
    st.header("Controls")
    if regions:
        selected_region = st.selectbox("Region", regions, index=0)
    else:
        selected_region = None
        st.selectbox("Region", ["No regions available"], index=0, disabled=True)
    history_days = st.slider("Actual history window", min_value=3, max_value=60, value=14, step=1)
    st.divider()
    st.caption("Input files")
    st.code(
        "\n".join(
            [
                str(ACTUAL_PATH.relative_to(ROOT)),
                str(FORECAST_PATH.relative_to(ROOT)),
                str(BACKTEST_FORECAST_PATH.relative_to(ROOT)),
                str(METRICS_PATH.relative_to(ROOT)),
            ]
        )
    )

latest_actual = actuals["period"].max() if not actuals.empty else None
latest_issue = forecasts["issue_time"].max() if not forecasts.empty else None
latest_target = forecasts["target_time"].max() if not forecasts.empty else None
status, status_message = freshness_status(latest_actual)

status_col, actual_col, forecast_col, horizon_col = st.columns(4)
status_col.metric("Data Status", status)
actual_col.metric("Latest Actual", format_time(latest_actual))
forecast_col.metric("Latest Forecast Run", format_time(latest_issue))
if latest_issue is not None and latest_target is not None and not pd.isna(latest_issue) and not pd.isna(latest_target):
    horizon_hours = int((latest_target - latest_issue).total_seconds() / 3600)
else:
    horizon_hours = None
horizon_col.metric("Forecast Horizon", f"{horizon_hours} hours" if horizon_hours is not None else "Unavailable")

show_status_message(status, status_message)

metric_cols = st.columns(4)
if not overall_metrics.empty:
    row = overall_metrics.iloc[0]
    metric_cols[0].metric("MAE", format_number(row.get("mae")))
    metric_cols[1].metric("RMSE", format_number(row.get("rmse")))
    metric_cols[2].metric("MAPE", format_percent(row.get("mape")))
    metric_cols[3].metric("sMAPE", format_percent(row.get("smape")))
else:
    for col, label in zip(metric_cols, ["MAE", "RMSE", "MAPE", "sMAPE"], strict=True):
        col.metric(label, "Unavailable")

if selected_region:
    st.subheader(f"{selected_region} Actuals and Forecast")
    chart_df = build_actual_forecast_chart(actuals, forecasts, selected_region, history_days)
    if chart_df.empty:
        st.warning("No chart data is available for the selected region.")
    else:
        st.line_chart(chart_df, x="time", y="mw", color="type", height=420)

    region_metrics = series_metrics.loc[series_metrics["region"] == selected_region] if not series_metrics.empty else pd.DataFrame()
    if not region_metrics.empty:
        region_row = region_metrics.iloc[0]
        st.subheader(f"{selected_region} Backtest Metrics")
        st.dataframe(
            region_metrics[["region", "series", "n", "mae", "rmse", "mape", "smape"]],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Latest regional backtest: "
            f"MAE {format_number(region_row.get('mae'))}, "
            f"RMSE {format_number(region_row.get('rmse'))}, "
            f"MAPE {format_percent(region_row.get('mape'))}."
        )

    st.subheader(f"{selected_region} Backtest: Forecast vs Actual")
    if backtest_forecasts.empty:
        st.info("No backtest forecast file found. Run `python -m src.jobs.run_train_and_forecast` to generate it.")
    else:
        overlay_df = build_backtest_overlay_chart(actuals, backtest_forecasts, selected_region)
        if overlay_df.empty:
            st.warning("No backtest forecast data available for this region.")
        else:
            bt_start = backtest_forecasts.loc[backtest_forecasts["region"] == selected_region, "target_time"].min()
            bt_end = backtest_forecasts.loc[backtest_forecasts["region"] == selected_region, "target_time"].max()
            st.caption(
                f"Holdout window: {format_time(bt_start)} → {format_time(bt_end)}  "
                f"({int((bt_end - bt_start).total_seconds() / 3600) + 1} hours). "
                "The vertical gap marks where history ends and the backtest predictions begin."
            )
            st.line_chart(overlay_df, x="time", y="mw", color="type", height=380)

st.subheader("Performance by Region")
if series_metrics.empty:
    st.warning("No per-region backtest metrics are available yet.")
else:
    sorted_metrics = series_metrics.sort_values("mape", ascending=False)
    st.bar_chart(sorted_metrics, x="region", y="mape", height=320)
    st.dataframe(
        sorted_metrics[["region", "series", "n", "mae", "rmse", "mape", "smape"]],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Data Preview", expanded=False):
    preview_cols = st.columns(2)
    with preview_cols[0]:
        st.markdown("**Recent actuals**")
        if actuals.empty:
            st.write("No actuals loaded.")
        else:
            st.dataframe(actuals.sort_values("period", ascending=False).head(50), use_container_width=True, hide_index=True)
    with preview_cols[1]:
        st.markdown("**Latest forecast**")
        if forecasts.empty:
            st.write("No forecasts loaded.")
        else:
            st.dataframe(forecasts.sort_values("target_time").head(50), use_container_width=True, hide_index=True)

with st.expander("Raw Backtest Report", expanded=False):
    if metrics_text:
        st.text(metrics_text)
    else:
        st.write("No backtest report loaded.")
