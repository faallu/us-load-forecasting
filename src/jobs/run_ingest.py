from __future__ import annotations

import logging

import pandas as pd

from src import storage
from src.config import Settings, get_settings
from src.logging_config import setup_logging
from src.pipeline.eia_client import DAILY_SUB_BA_ENDPOINT, SUB_BA_ENDPOINT, EIAClient, EIAEndpoint
from src.pipeline.state_store import StateStore
from src.pipeline.transform import deduplicate_latest, transform_subba_to_region_total

LOGGER = logging.getLogger(__name__)

# Re-pull a safety window of recent days on every daily ingest to catch EIA revisions.
DAILY_BACKFILL_DAYS = 7


def _read_parquet_if_exists(path: str) -> pd.DataFrame:
    if storage.exists(path):
        return storage.read_parquet(path)
    return pd.DataFrame()


def _target_pull_start(
    watermark: pd.Timestamp | None,
    default_start: str,
    backfill: pd.Timedelta,
) -> pd.Timestamp:
    if watermark is None:
        return pd.to_datetime(default_start, utc=True)
    return watermark - backfill


def _ingest_dataset(
    *,
    client: EIAClient,
    store: StateStore,
    endpoint: EIAEndpoint,
    dataset_name: str,
    processed_path: str,
    settings: Settings,
    start: pd.Timestamp,
    end: pd.Timestamp,
    extra_facets: list[tuple[str, str]] | None = None,
) -> int:
    raw_subba = client.fetch_with_pagination_and_chunking(
        endpoint=endpoint,
        regions=list(settings.allowed_regions),
        start=start,
        end=end,
        extra_facets=extra_facets,
    )
    new_rows = transform_subba_to_region_total(raw_subba, settings.allowed_regions)
    existing = _read_parquet_if_exists(processed_path)
    merged = deduplicate_latest(
        pd.concat([existing, new_rows], ignore_index=True), ["region", "series", "period"]
    )
    storage.write_parquet(merged, processed_path)
    store.update_dataset_state(dataset_name=dataset_name, df=merged, partition_cols=["region", "series"])
    LOGGER.info("Ingest complete: dataset=%s rows=%s -> %s", dataset_name, len(merged), processed_path)
    return len(merged)


def run_ingest() -> None:
    setup_logging()
    settings = get_settings()

    if not settings.eia_api_key:
        raise ValueError("EIA_API_KEY is required. Set it in environment variables or .env.")

    store = StateStore(settings.state_path)
    client = EIAClient(base_url=settings.eia_base_url, api_key=settings.eia_api_key)
    now = pd.Timestamp.now(tz="UTC")

    # Hourly demand (7d / 14d / 1mo windows).
    hourly_start = _target_pull_start(
        watermark=store.get_dataset_watermark("load"),
        default_start=settings.default_history_start,
        backfill=pd.Timedelta(hours=settings.pull_backfill_hours),
    )
    _ingest_dataset(
        client=client,
        store=store,
        endpoint=SUB_BA_ENDPOINT,
        dataset_name="load",
        processed_path=settings.processed_load_path,
        settings=settings,
        start=hourly_start,
        end=now,
    )

    # Daily demand (3mo / 6mo / 1yr windows). EIA's daily value is the SUM of hourly
    # demand (MWh); a single timezone facet selects the day-boundary convention so we
    # do not ingest the five redundant timezone copies.
    daily_start = _target_pull_start(
        watermark=store.get_dataset_watermark("load_daily"),
        default_start=settings.default_history_start,
        backfill=pd.Timedelta(days=DAILY_BACKFILL_DAYS),
    )
    _ingest_dataset(
        client=client,
        store=store,
        endpoint=DAILY_SUB_BA_ENDPOINT,
        dataset_name="load_daily",
        processed_path=settings.processed_load_daily_path,
        settings=settings,
        start=daily_start,
        end=now,
        extra_facets=[("timezone", settings.eia_daily_timezone)],
    )


if __name__ == "__main__":
    run_ingest()
