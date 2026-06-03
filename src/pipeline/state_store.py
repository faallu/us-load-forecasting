from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src import storage


class StateStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = str(path)

    def read(self) -> dict[str, Any]:
        if not storage.exists(self.path):
            return {}
        return storage.read_json(self.path)

    def write(self, state: dict[str, Any]) -> None:
        storage.write_json(self.path, state)

    def get_dataset_watermark(self, dataset_name: str) -> pd.Timestamp | None:
        state = self.read()
        raw = state.get(dataset_name, {}).get("global_max_period")
        if not raw:
            return None
        return pd.to_datetime(raw, utc=True)

    def update_dataset_state(
        self,
        dataset_name: str,
        df: pd.DataFrame,
        partition_cols: list[str],
    ) -> None:
        state = self.read()
        ds_state = state.get(dataset_name, {})

        if df.empty:
            state[dataset_name] = ds_state
            self.write(state)
            return

        df_local = df.copy()
        df_local["period"] = pd.to_datetime(df_local["period"], utc=True)

        global_max = df_local["period"].max()
        ds_state["global_max_period"] = global_max.isoformat()

        partition_watermarks: dict[str, str] = {}
        grouped = df_local.groupby(partition_cols)["period"].max().reset_index()
        for _, row in grouped.iterrows():
            key = "|".join(str(row[col]) for col in partition_cols)
            partition_watermarks[key] = pd.Timestamp(row["period"]).isoformat()
        ds_state["partition_max_period"] = partition_watermarks

        state[dataset_name] = ds_state
        self.write(state)
