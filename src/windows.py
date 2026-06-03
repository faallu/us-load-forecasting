"""Forecast window registry: the single source of truth for the deployed windows.

Each window defines how many steps to forecast and at what resolution. Short
windows run at hourly resolution; long windows run at daily resolution because a
recursive hourly forecast over thousands of steps is both slow and error-prone.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastWindow:
    key: str
    label: str
    horizon: int  # number of steps at the window's resolution
    resolution: str  # "hourly" or "daily"

    @property
    def freq(self) -> str:
        return "h" if self.resolution == "hourly" else "D"

    @property
    def horizon_hours(self) -> int:
        """Approximate horizon expressed in hours (for display/comparison)."""
        return self.horizon if self.resolution == "hourly" else self.horizon * 24


FORECAST_WINDOWS: dict[str, ForecastWindow] = {
    "7d": ForecastWindow("7d", "7 days", 168, "hourly"),
    "14d": ForecastWindow("14d", "14 days", 336, "hourly"),
    "1mo": ForecastWindow("1mo", "1 month", 720, "hourly"),
    "3mo": ForecastWindow("3mo", "3 months", 90, "daily"),
    "6mo": ForecastWindow("6mo", "6 months", 180, "daily"),
    "1yr": ForecastWindow("1yr", "1 year", 365, "daily"),
}

# Window also written to the legacy non-windowed XGBoost paths so the existing
# Streamlit dashboard keeps working without changes.
DEFAULT_WINDOW = "7d"


def get_window(key: str) -> ForecastWindow:
    try:
        return FORECAST_WINDOWS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(FORECAST_WINDOWS))
        raise KeyError(f"Unknown forecast window '{key}'. Valid windows: {valid}") from exc


def all_windows() -> list[ForecastWindow]:
    return list(FORECAST_WINDOWS.values())


def window_keys() -> list[str]:
    return list(FORECAST_WINDOWS.keys())
