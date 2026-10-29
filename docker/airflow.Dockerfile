FROM apache/airflow:2.10.5-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/project/airflow/dags

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/project

COPY requirements.txt ./requirements.txt
COPY --chown=airflow:root airflow/dags ./airflow/dags
COPY --chown=airflow:root src ./src
COPY --chown=airflow:root configs ./configs

USER airflow

RUN python -m pip install \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

EXPOSE 8080
