from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EIAEndpoint:
    path: str
    facet_name: str
    value_column: str = "value"
    period_column: str = "period"
    frequency: str = "hourly"

    @property
    def period_format(self) -> str:
        return "%Y-%m-%d" if self.frequency == "daily" else "%Y-%m-%dT%H"


# Hourly demand by sub-BA (used for the 7d/14d/1mo windows).
SUB_BA_ENDPOINT = EIAEndpoint(
    path="/electricity/rto/region-sub-ba-data/data/",
    facet_name="parent",
    frequency="hourly",
)

# Daily demand by sub-BA (used for the 3mo/6mo/1yr windows). EIA's daily ``value`` is
# the SUM of hourly demand (MWh), so it mirrors aggregating the hourly series. The
# endpoint repeats each sub-BA once per ``timezone`` (the day-boundary timezone), so a
# single timezone facet must be supplied to avoid double counting.
DAILY_SUB_BA_ENDPOINT = EIAEndpoint(
    path="/electricity/rto/daily-region-sub-ba-data/data/",
    facet_name="parent",
    frequency="daily",
)


class EIAClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _params_for_request(
        self,
        endpoint: EIAEndpoint,
        regions: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int,
        length: int,
        extra_facets: list[tuple[str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = [
            ("api_key", self.api_key),
            ("frequency", endpoint.frequency),
            ("start", start.strftime(endpoint.period_format)),
            ("end", end.strftime(endpoint.period_format)),
            ("sort[0][column]", endpoint.period_column),
            ("sort[0][direction]", "asc"),
            ("offset", str(offset)),
            ("length", str(length)),
            ("data[0]", endpoint.value_column),
        ]
        for region in regions:
            params.append((f"facets[{endpoint.facet_name}][]", region))
        for facet_name, facet_value in extra_facets or []:
            params.append((f"facets[{facet_name}][]", facet_value))
        return params

    def _call_api(self, endpoint: EIAEndpoint, params: list[tuple[str, str]]) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint.path}"
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json().get("response", {})
            except requests.RequestException as exc:
                last_error = exc
                wait_seconds = 2**attempt
                LOGGER.warning("EIA request failed (attempt=%s): %s", attempt + 1, exc)
                if attempt < 3:
                    time.sleep(wait_seconds)
        raise RuntimeError(f"EIA API request failed after retries: {last_error}") from last_error

    def _fetch_window(
        self,
        endpoint: EIAEndpoint,
        regions: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        page_size: int = 5000,
        extra_facets: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        offset = 0
        rows: list[dict[str, Any]] = []
        total = None

        while True:
            params = self._params_for_request(
                endpoint=endpoint,
                regions=regions,
                start=start,
                end=end,
                offset=offset,
                length=page_size,
                extra_facets=extra_facets,
            )
            payload = self._call_api(endpoint, params)
            batch = payload.get("data", [])
            if total is None:
                total = payload.get("total")

            if not batch:
                break

            rows.extend(batch)
            offset += len(batch)

            if total is not None and offset >= int(total):
                break
            if len(batch) < page_size:
                break

        return pd.DataFrame(rows)

    def fetch_with_pagination_and_chunking(
        self,
        endpoint: EIAEndpoint,
        regions: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        extra_facets: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Bypass API row limits by combining pagination and monthly date chunking."""
        frames: list[pd.DataFrame] = []
        cursor = pd.Timestamp(start).tz_convert("UTC") if start.tzinfo else pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC") if end.tzinfo else pd.Timestamp(end, tz="UTC")

        while cursor <= end_ts:
            window_end = min((cursor + pd.DateOffset(months=1) - timedelta(hours=1)), end_ts)
            window_df = self._fetch_window(
                endpoint=endpoint,
                regions=regions,
                start=cursor,
                end=window_end,
                extra_facets=extra_facets,
            )
            if not window_df.empty:
                frames.append(window_df)
            cursor = window_end + timedelta(hours=1)

        if not frames:
            return pd.DataFrame()

        output = pd.concat(frames, ignore_index=True)
        output = output.drop_duplicates()
        LOGGER.info(
            "Fetched %s rows from %s between %s and %s",
            len(output),
            endpoint.path,
            start,
            end,
        )
        return output
