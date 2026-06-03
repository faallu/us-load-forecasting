"""Hourly EIA ingest DAG.

Pulls new EIA load data and merges it into the processed parquet in S3.
"""

from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

DEFAULT_ARGS = {"retries": 2, "retry_delay": pendulum.duration(minutes=5)}


@dag(
    dag_id="load_ingest",
    schedule="@hourly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["load-forecasting", "ingest"],
)
def load_ingest():
    @task
    def ingest() -> None:
        from src.jobs.run_ingest import run_ingest

        run_ingest()

    ingest()


load_ingest()
