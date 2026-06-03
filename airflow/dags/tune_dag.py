"""Weekly grid-search DAG.

Grid search is expensive (param combos x series x recursive horizon), so it runs
weekly rather than on every forecast. One task per window writes the winning
hyperparameters to ``models/best_params/{window}.json`` in S3, which the daily
forecast DAG then reuses.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

from src.windows import window_keys

DEFAULT_ARGS = {"retries": 1, "retry_delay": pendulum.duration(minutes=10)}


@dag(
    dag_id="load_tune",
    schedule="@weekly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["load-forecasting", "tuning"],
)
def load_tune():
    @task
    def tune(window_key: str) -> str:
        from src.jobs.run_train_and_forecast import run_tuning

        run_tuning([window_key])
        return window_key

    # One mapped task per window so they run in parallel and fail independently.
    tune.expand(window_key=window_keys())


load_tune()
