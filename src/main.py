from src.jobs.run_ingest import run_ingest
from src.jobs.run_train_and_forecast import run_train_and_forecast
from src.scheduler import run_scheduler


def main() -> None:
    run_ingest()
    run_train_and_forecast()


if __name__ == "__main__":
    # Initial bootstrap run before switching to background automation.
    main()
    run_scheduler()
