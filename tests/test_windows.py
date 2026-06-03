import pytest

from src.windows import FORECAST_WINDOWS, all_windows, get_window, window_keys


def test_registry_has_expected_windows() -> None:
    assert window_keys() == ["7d", "14d", "1mo", "3mo", "6mo", "1yr"]
    assert len(all_windows()) == 6


def test_resolution_split_short_hourly_long_daily() -> None:
    hourly = {w.key for w in all_windows() if w.resolution == "hourly"}
    daily = {w.key for w in all_windows() if w.resolution == "daily"}
    assert hourly == {"7d", "14d", "1mo"}
    assert daily == {"3mo", "6mo", "1yr"}


def test_horizon_hours_conversion() -> None:
    assert get_window("7d").horizon_hours == 168
    assert get_window("3mo").horizon_hours == 90 * 24
    assert get_window("3mo").freq == "D"
    assert get_window("7d").freq == "h"


def test_unknown_window_raises() -> None:
    with pytest.raises(KeyError):
        get_window("nope")
    assert "1yr" in FORECAST_WINDOWS
