# Load Forecasting

Automated EIA regional hourly load forecasting with **XGBoost**, served on AWS.

- Pulls EIA region sub-BA data and aggregates it to `"<REGION> total"`.
- Produces forecasts across **six windows**: `7d`, `14d`, `1mo` (hourly resolution) and `3mo`, `6mo`, `1yr` (daily resolution).
- **Grid-searches** the best XGBoost hyperparameters per window.
- Stores all data/artifacts in **S3** (or locally for dev), orchestrates with **Airflow**, and serves precomputed forecasts via **FastAPI on AWS Lambda**.

## Architecture

```
EIA API ──> Airflow (ingest / weekly tune / daily forecast) ──> S3 ──> FastAPI on Lambda ──> clients
```

- **Ingest** (hourly): pulls new EIA load, merges into `processed/load_hourly.parquet`.
- **Tune** (weekly): grid search per window, writes `models/best_params/{window}.json`.
- **Forecast** (daily): trains XGBoost per window using cached best params, writes per-window forecast/metrics/plot artifacts.
- **API** (Lambda): reads precomputed artifacts from S3 and serves them as JSON. It never trains and does not depend on xgboost, keeping the image small.

Short windows run at hourly resolution; long windows run at daily resolution. Rather than aggregating hourly data in code, daily windows pull EIA's **daily** demand series directly (the daily `value` is the SUM of hourly demand in MWh, so it mirrors aggregating the hourly series). A recursive hourly forecast over thousands of steps would be slow and accumulate error.

## Data Sources

- [EIA region sub-BA data browser](https://www.eia.gov/opendata/browser/electricity/rto/region-sub-ba-data?frequency=hourly&data=value;&facets=parent;&parent=CISO;ERCO;ISNE;MISO;NYIS;PJM;SWPP;&sortColumn=period;&sortDirection=desc;) (hourly).
- [EIA daily region sub-BA data browser](https://www.eia.gov/opendata/browser/electricity/rto/daily-region-sub-ba-data) (daily).

The browser links are UI wrappers; the implementation calls EIA API v2 endpoints directly with pagination + monthly date chunking and incremental high-watermark pulls. The daily endpoint returns each sub-BA once per day-boundary `timezone`; a single timezone (`EIA_DAILY_TIMEZONE`, default `Eastern`) is kept so per-region daily totals are not double counted.

## Project Layout

- `src/storage.py` - local/S3 storage abstraction (fsspec/s3fs).
- `src/windows.py` - forecast window registry (horizon + resolution).
- `src/config.py` - settings; S3-aware paths via `STORAGE_BACKEND`.
- `src/pipeline/` - EIA client, transforms, incremental ingest state.
- `src/jobs/run_ingest.py` - ingest job.
- `src/jobs/run_train_and_forecast.py` - multi-window train/forecast/backtest + tuning entry points.
- `src/models/xgboost_model.py` - hourly + daily XGBoost feature engineering and forecasting.
- `src/models/tuning.py` - per-window grid search.
- `src/models/backtest.py`, `src/models/evaluate.py` - backtesting and metrics.
- `airflow/` - Dockerfile, docker-compose, and the three DAGs.
- `api/` - FastAPI app, Lambda Dockerfile, serve-only requirements.
- `deploy/` - AWS deployment guide, IAM policies, image build/push script.

> SARIMA (`src/models/sarima.py`) and the Streamlit dashboards remain in the repo for local use but are not part of the deployed pipeline.

## Setup (local)

1. Create and activate a Python environment.
2. Install dependencies: `pip install -e .[dev]` (or `pip install -r requirements.txt`).
3. Copy `.env.example` to `.env` and set `EIA_API_KEY`. Leave `STORAGE_BACKEND=local` to write under `./data`.

## Run (local)

```bash
python -m src.jobs.run_ingest                         # pull EIA data
python -m src.jobs.run_train_and_forecast --tune-only # grid search all windows -> best-params JSON
python -m src.jobs.run_train_and_forecast             # forecast + backtest all windows
python -m src.jobs.run_train_and_forecast --window 7d # a single window
uvicorn api.main:app --reload                         # serve the API locally (http://127.0.0.1:8000)
pytest                                                # tests
```

Useful flags: `--window KEY` (repeatable), `--tune-only`, `--no-tune` (use model defaults instead of grid search when a window is untuned).

## Storage layout

Under `./data` (local) or `s3://$S3_BUCKET/$S3_PREFIX` (S3):

```
state/ingestion_state.json
processed/load_hourly.parquet                      # hourly demand (7d/14d/1mo)
processed/load_daily.parquet                       # daily demand MWh (3mo/6mo/1yr)
models/best_params/{window}.json
forecasts/load_forecast_xgb_{window}.parquet
forecasts/load_backtest_forecast_xgb_{window}.parquet
metrics/load_backtest_metrics_xgb_{window}.{txt,parquet}
reports/load_backtest_xgb_{window}.png
```

## API endpoints

- `GET /health`
- `GET /windows`
- `GET /regions`
- `GET /forecast?window=7d&region=CISO`
- `GET /metrics?window=7d`

## AWS deployment

See [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md) for S3 setup, ECR images, IAM, self-hosted Airflow on EC2/ECS, and Lambda + API Gateway.
