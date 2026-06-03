import importlib

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point the storage layer at a temp local dir and reload config + api so the
    # fresh environment is picked up.
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    from src.config import get_settings

    settings = get_settings()
    forecast = pd.DataFrame(
        {
            "region": ["CISO", "CISO", "PJM"],
            "series": ["CISO total", "CISO total", "PJM total"],
            "issue_time": pd.to_datetime(["2024-01-01T00:00:00Z"] * 3, utc=True),
            "target_time": pd.to_datetime(
                ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z", "2024-01-01T01:00:00Z"], utc=True
            ),
            "yhat": [100.0, 101.0, 200.0],
        }
    )
    from src import storage

    storage.write_parquet(forecast, settings.forecast_path("7d"))

    api_main = importlib.import_module("api.main")
    api_main._cache.clear()
    return TestClient(api_main.app)


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_windows_and_regions(client) -> None:
    windows = client.get("/windows").json()["windows"]
    assert {w["key"] for w in windows} == {"7d", "14d", "1mo", "3mo", "6mo", "1yr"}
    regions = client.get("/regions").json()["regions"]
    assert "CISO" in regions


def test_forecast_filtered_by_region(client) -> None:
    resp = client.get("/forecast", params={"window": "7d", "region": "CISO"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["window"] == "7d"
    assert body["count"] == 2
    assert all(row["region"] == "CISO" for row in body["forecast"])


def test_forecast_unknown_window_is_400(client) -> None:
    resp = client.get("/forecast", params={"window": "bogus"})
    assert resp.status_code == 400


def test_forecast_missing_window_is_404(client) -> None:
    resp = client.get("/forecast", params={"window": "1yr"})
    assert resp.status_code == 404
