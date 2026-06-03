from pathlib import Path

import pandas as pd

from src import storage


def test_join_local_and_remote() -> None:
    assert storage.join("data", "forecasts", "x.parquet") == "data/forecasts/x.parquet"
    assert storage.join("s3://bucket/prefix", "metrics", "m.txt") == "s3://bucket/prefix/metrics/m.txt"
    assert storage.is_remote("s3://bucket/x") is True
    assert storage.is_remote("data/x") is False


def test_parquet_text_json_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    parquet_path = str(tmp_path / "nested" / "df.parquet")
    storage.write_parquet(df, parquet_path)
    assert storage.exists(parquet_path)
    pd.testing.assert_frame_equal(storage.read_parquet(parquet_path), df)

    text_path = str(tmp_path / "nested" / "note.txt")
    storage.write_text(text_path, "hello")
    assert storage.read_text(text_path) == "hello"

    json_path = str(tmp_path / "nested" / "params.json")
    storage.write_json(json_path, {"n_estimators": 300, "learning_rate": 0.1})
    assert storage.read_json(json_path) == {"n_estimators": 300, "learning_rate": 0.1}


def test_exists_false_for_missing(tmp_path: Path) -> None:
    assert storage.exists(str(tmp_path / "missing.parquet")) is False
