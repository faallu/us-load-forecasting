"""Daily forecast DAG.

For each window, trains XGBoost on all available history using the cached best
params (from the weekly tune DAG), produces the forward forecast, runs the backtest,
and writes per-window artifacts to S3. Uses model defaults if a window has not been
tuned yet (``tune_if_missing=False`` keeps this DAG fast and predictable).
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.windows import window_keys

DEFAULT_ARGS = {"retries": 1, "retry_delay": pendulum.duration(minutes=10)}


@dag(
    dag_id="load_forecast",
    schedule="@daily",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["load-forecasting", "forecast"],
)
def load_forecast():
    @task
    def forecast(window_key: str) -> str:
        from src.jobs.run_train_and_forecast import run_train_and_forecast

        run_train_and_forecast([window_key], tune_if_missing=False)
        return window_key

    forecast.expand(window_key=window_keys())


load_forecast()
