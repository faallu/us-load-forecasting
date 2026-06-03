# AWS Deployment Guide

This deploys the XGBoost-only load-forecasting pipeline:

- **S3** stores all data and artifacts.
- **Airflow** (self-hosted Docker on EC2/ECS) orchestrates ingest, weekly tuning, and daily forecasting.
- **FastAPI on Lambda** (container image) serves precomputed forecasts/metrics read from S3.

```
EIA API ──> Airflow (ingest/tune/forecast) ──> S3 ──> Lambda (FastAPI) ──> clients
```

## 1. S3 bucket and layout

Create one bucket and pick a prefix (e.g. `load-forecasting`):

```bash
aws s3 mb s3://YOUR_BUCKET --region us-east-1
```

Artifacts the pipeline writes (under `s3://YOUR_BUCKET/load-forecasting/`):

```
state/ingestion_state.json                         # ingest watermarks (load + load_daily)
processed/load_hourly.parquet                      # hourly actuals (7d/14d/1mo)
processed/load_daily.parquet                       # daily actuals MWh (3mo/6mo/1yr)
models/best_params/{window}.json                   # tuned hyperparameters per window
forecasts/load_forecast_xgb_{window}.parquet       # forward forecast per window
forecasts/load_backtest_forecast_xgb_{window}.parquet
metrics/load_backtest_metrics_xgb_{window}.txt     # human-readable report
metrics/load_backtest_metrics_xgb_{window}.parquet # per-series metrics
reports/load_backtest_xgb_{window}.png             # actual vs forecast plot
```

Windows: `7d`, `14d`, `1mo` (hourly) and `3mo`, `6mo`, `1yr` (daily).

## 2. ECR repositories and images

```bash
AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1 ./deploy/build_and_push.sh
```

This creates `load-forecasting-airflow` and `load-forecasting-api` ECR repos and pushes both images. (Build from the repo root; both Dockerfiles `COPY src` from there.)

## 3. IAM

- **Airflow** (EC2 instance profile or ECS task role): attach `deploy/iam-airflow-policy.json` (S3 read/write). Replace `REPLACE_WITH_BUCKET`.
- **Lambda** (execution role): attach `deploy/iam-lambda-policy.json` (S3 read-only) plus the AWS-managed `AWSLambdaBasicExecutionRole` for CloudWatch logs.

## 4. Airflow on EC2 (self-hosted Docker)

1. Launch an EC2 instance (e.g. `t3.large`+, Amazon Linux 2023) with the Airflow IAM instance profile and a security group allowing your IP to reach port 8080.
2. Install Docker + the compose plugin, then pull the image and bring up the stack:

```bash
# on the instance, in a checkout of this repo
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/load-forecasting-airflow:latest
docker tag  123456789012.dkr.ecr.us-east-1.amazonaws.com/load-forecasting-airflow:latest load-forecasting-airflow:latest

cd airflow
cp .env.example .env        # set S3_BUCKET, S3_PREFIX, EIA_API_KEY; leave AWS keys blank (use the instance role)
docker compose up airflow-init
docker compose up -d
```

3. Open `http://EC2_PUBLIC_IP:8080`, unpause the `load_ingest`, `load_tune`, and `load_forecast` DAGs.
   - `load_ingest` runs hourly.
   - `load_tune` runs weekly (grid search per window -> best-params JSON in S3).
   - `load_forecast` runs daily (uses cached best params; model defaults if a window is untuned).

> ECS alternative: register the same image as a task definition with the Airflow task role, run the scheduler/webserver as services, and use RDS Postgres for the metadata DB instead of the bundled `postgres` container.

## 5. Lambda + API Gateway

Create the function from the API image and wire an HTTP API:

```bash
ACCOUNT=123456789012; REGION=us-east-1; BUCKET=YOUR_BUCKET; PREFIX=load-forecasting
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/load-forecasting-api:latest

aws lambda create-function \
  --function-name load-forecasting-api \
  --package-type Image \
  --code ImageUri=$IMAGE \
  --role arn:aws:iam::$ACCOUNT:role/load-forecasting-lambda-role \
  --timeout 30 --memory-size 1024 \
  --environment "Variables={STORAGE_BACKEND=s3,S3_BUCKET=$BUCKET,S3_PREFIX=$PREFIX,AWS_DEFAULT_REGION=$REGION}" \
  --region $REGION

# Simplest public endpoint: a Function URL
aws lambda create-function-url-config \
  --function-name load-forecasting-api --auth-type NONE --region $REGION
```

For a managed REST surface use API Gateway (HTTP API) with a Lambda proxy integration to `load-forecasting-api` instead of the Function URL.

Update after pushing a new image:

```bash
aws lambda update-function-code --function-name load-forecasting-api --image-uri $IMAGE --region $REGION
```

### API endpoints

- `GET /health`
- `GET /windows`
- `GET /regions`
- `GET /forecast?window=7d&region=CISO`
- `GET /metrics?window=7d`

## 6. First-run order

1. Bring up Airflow and let `load_ingest` populate `processed/load_hourly.parquet`.
2. Trigger `load_tune` once (or wait for the weekly run) to write best-params JSON.
3. Trigger `load_forecast` to write per-window forecasts/metrics.
4. Hit the API to confirm forecasts are served.

You can also run any step manually from a shell with the project installed and env configured:

```bash
python -m src.jobs.run_ingest
python -m src.jobs.run_train_and_forecast --tune-only          # grid search all windows
python -m src.jobs.run_train_and_forecast --window 7d          # forecast a single window
python -m src.jobs.run_train_and_forecast                      # forecast all windows
```
