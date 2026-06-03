#!/usr/bin/env bash
# Build and push the Airflow and API images to ECR.
# Run from the REPO ROOT. Requires: awscli, docker, and an authenticated AWS session.
#
#   AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1 ./deploy/build_and_push.sh
set -euo pipefail

: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${AWS_REGION:=us-east-1}"

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
AIRFLOW_REPO="load-forecasting-airflow"
API_REPO="load-forecasting-api"

echo ">> Ensuring ECR repositories exist"
aws ecr describe-repositories --repository-names "${AIRFLOW_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${AIRFLOW_REPO}" --region "${AWS_REGION}" >/dev/null
aws ecr describe-repositories --repository-names "${API_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${API_REPO}" --region "${AWS_REGION}" >/dev/null

echo ">> Logging in to ECR ${REGISTRY}"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo ">> Building and pushing Airflow image"
docker build -f airflow/Dockerfile -t "${REGISTRY}/${AIRFLOW_REPO}:latest" .
docker push "${REGISTRY}/${AIRFLOW_REPO}:latest"

echo ">> Building and pushing API (Lambda) image"
docker build -f api/Dockerfile -t "${REGISTRY}/${API_REPO}:latest" .
docker push "${REGISTRY}/${API_REPO}:latest"

echo ">> Done."
echo "   Airflow image: ${REGISTRY}/${AIRFLOW_REPO}:latest"
echo "   API image    : ${REGISTRY}/${API_REPO}:latest"
