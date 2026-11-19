# syntax=docker/dockerfile:1
# Build a partir da raiz do projeto:
#   docker build -f docker/Dockerfile -t hospital-triage-ai:1.0.0 .
#
# O modelo é treinado durante o build (seed fixa), então a imagem não depende de
# nenhum artefato fora do git. Python 3.13 porque o numpy 2.5 e o scipy 1.18 do
# requirements.txt não têm wheel para 3.11.
ARG PYTHON_VERSION=3.13

# venv com as dependências
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --prefer-binary -r /tmp/requirements.txt

# gera os dados sintéticos e treina o modelo
FROM python:${PYTHON_VERSION}-slim AS trainer

# o XGBoost precisa do libgomp1 (OpenMP)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/venv /opt/venv
WORKDIR /build
COPY src ./src

ARG N_RECORDS=10000
ARG SEED=42
RUN python -m src.pipelines.generate_mimic_data --n-records "${N_RECORDS}" --seed "${SEED}" \
 && python -m src.pipelines.train_model --seed "${SEED}"

# imagem final
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="hospital-triage-ai" \
      org.opencontainers.image.description="API de triagem de risco de readmissão hospitalar em 30 dias (XGBoost + SHAP)" \
      org.opencontainers.image.version="1.0.0"

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 triage \
 && useradd --system --uid 10001 --gid triage --no-create-home --shell /usr/sbin/nologin triage

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRIAGE_ENVIRONMENT=production \
    TRIAGE_MODEL_PATH=/app/models/readmission_model.joblib

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=triage:triage src ./src
COPY --from=trainer --chown=triage:triage /build/models ./models

USER triage
EXPOSE 8000

# /health dá 200 em ok/degraded e 503 sem modelo
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# cada worker carrega a própria cópia do modelo e do explainer
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
