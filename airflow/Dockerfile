# Self-hosted Airflow image with the load-forecasting project baked in.
# Build from the REPO ROOT so the COPY paths resolve:
#   docker build -f airflow/Dockerfile -t load-forecasting-airflow .
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Project lives on PYTHONPATH so DAGs can `import src...`.
ENV PYTHONPATH=/opt/load-forecasting

# Install the forecasting dependencies (xgboost, pandas, s3fs, ...). Airflow itself
# is already pinned in the base image, so we install our extras on top.
COPY requirements.txt /tmp/load-forecasting-requirements.txt
RUN pip install --no-cache-dir -r /tmp/load-forecasting-requirements.txt

# Copy the application code and DAGs into the image.
COPY src /opt/load-forecasting/src
COPY pyproject.toml /opt/load-forecasting/pyproject.toml
COPY airflow/dags /opt/airflow/dags
