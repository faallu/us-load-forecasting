#!/bin/bash
set -euxo pipefail
exec > /var/log/load-forecasting-bootstrap.log 2>&1

REGION="us-east-2"
ACCOUNT="888577025527"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
APP_DIR="/opt/load-forecasting/airflow"

dnf update -y
dnf install -y docker aws-cli
systemctl enable --now docker

mkdir -p /usr/local/lib/docker/cli-plugins
COMPOSE_VER="v2.32.4"
curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VER}/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
docker pull "${REGISTRY}/load-forecasting-airflow:latest"
docker tag "${REGISTRY}/load-forecasting-airflow:latest" load-forecasting-airflow:latest

EIA_API_KEY="$(aws ssm get-parameter \
  --name /load-forecasting/eia-api-key \
  --with-decryption \
  --region "${REGION}" \
  --query Parameter.Value \
  --output text)"

mkdir -p "${APP_DIR}"
cat > "${APP_DIR}/.env" <<EOF
STORAGE_BACKEND=s3
S3_BUCKET=eia-load-forecasting
S3_PREFIX=
AWS_DEFAULT_REGION=${REGION}
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
EIA_API_KEY=${EIA_API_KEY}
EIA_BASE_URL=https://api.eia.gov/v2
EIA_DAILY_TIMEZONE=Eastern
DEFAULT_HISTORY_START=2023-01-01T00:00:00Z
XGB_TEST_FRACTION=0.2
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=admin
EOF

cat > "${APP_DIR}/docker-compose.yaml" <<'COMPOSE'
x-airflow-common: &airflow-common
  image: load-forecasting-airflow:latest
  environment: &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: "true"
    PYTHONPATH: /opt/load-forecasting
    STORAGE_BACKEND: ${STORAGE_BACKEND:-s3}
    S3_BUCKET: ${S3_BUCKET:-}
    S3_PREFIX: ${S3_PREFIX:-}
    AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION:-us-east-2}
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:-}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:-}
    EIA_API_KEY: ${EIA_API_KEY:-}
    EIA_BASE_URL: ${EIA_BASE_URL:-https://api.eia.gov/v2}
    EIA_DAILY_TIMEZONE: ${EIA_DAILY_TIMEZONE:-Eastern}
    DEFAULT_HISTORY_START: ${DEFAULT_HISTORY_START:-2023-01-01T00:00:00Z}
    FORECAST_MODEL: xgboost
    XGB_TEST_FRACTION: ${XGB_TEST_FRACTION:-0.2}
  volumes:
    - airflow-logs:/opt/airflow/logs
  depends_on: &airflow-common-depends-on
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5
    restart: always

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db migrate
        airflow users create \
          --username "${AIRFLOW_ADMIN_USER:-admin}" \
          --password "${AIRFLOW_ADMIN_PASSWORD:-admin}" \
          --firstname Admin --lastname User --role Admin \
          --email admin@example.com || true
    restart: on-failure

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    restart: always

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    restart: always

volumes:
  postgres-db:
  airflow-logs:
COMPOSE

cd "${APP_DIR}"
docker compose up airflow-init
docker compose up -d
echo "Airflow stack started."
