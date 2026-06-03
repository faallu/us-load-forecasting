from pathlib import Path

import pandas as pd

from src.pipeline.state_store import StateStore
from src.pipeline.transform import deduplicate_latest


def test_state_store_tracks_global_and_partition_watermarks(tmp_path: Path) -> None:
    state_path = tmp_path / "ingestion_state.json"
    store = StateStore(state_path)
    df = pd.DataFrame(
        {
            "region": ["CISO", "CISO", "PJM"],
            "series": ["CISO total", "CISO total", "PJM total"],
            "period": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T00:00:00Z"],
                utc=True,
            ),
            "value": [1.0, 2.0, 3.0],
        }
    )

    store.update_dataset_state("load", df, ["region", "series"])
    wm = store.get_dataset_watermark("load")
    assert wm == pd.Timestamp("2026-01-01T01:00:00Z")


def test_deduplicate_latest_keeps_last_record() -> None:
    raw = pd.DataFrame(
        {
            "region": ["CISO", "CISO"],
            "series": ["CISO total", "CISO total"],
            "period": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], utc=True),
            "value": [10.0, 12.0],
        }
    )
    out = deduplicate_latest(raw, ["region", "series", "period"])
    assert len(out) == 1
    assert float(out.iloc[0]["value"]) == 12.0
