from __future__ import annotations

import logging
import time

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import get_settings
from src.jobs.run_ingest import run_ingest
from src.jobs.run_train_and_forecast import run_train_and_forecast
from src.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)


def run_scheduler() -> None:
    setup_logging()
    settings = get_settings()

    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingest, "interval", minutes=settings.ingest_interval_minutes, id="hourly_ingest")
    scheduler.add_job(
        run_train_and_forecast,
        "interval",
        hours=settings.train_interval_hours,
        id="daily_train_forecast",
    )
    scheduler.start()
    LOGGER.info("Scheduler started (ingest every %s min, train every %s h)", settings.ingest_interval_minutes, settings.train_interval_hours)

    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        scheduler.shutdown()
        LOGGER.info("Scheduler stopped.")
