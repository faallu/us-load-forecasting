# FastAPI-on-Lambda container image (serve-only).
# Build from the REPO ROOT so the COPY paths resolve:
#   docker build -f api/Dockerfile -t load-forecasting-api .
FROM public.ecr.aws/lambda/python:3.13

# Install serve-only dependencies into the Lambda task root.
COPY api/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Application code: the API plus the shared config/storage/windows modules.
COPY api ${LAMBDA_TASK_ROOT}/api
COPY src ${LAMBDA_TASK_ROOT}/src

# Default storage backend for the API (override per-environment in Lambda config).
ENV STORAGE_BACKEND=s3

# Mangum handler: module path "api.main", attribute "handler".
CMD ["api.main.handler"]
