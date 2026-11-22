# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src/aegis ./src/aegis

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir ".[runtime]" \
    && find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} +

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_WAIT_POLICY=PASSIVE \
    AEGIS_HOST=0.0.0.0 \
    AEGIS_PORT=8000 \
    AEGIS_MODE=enforce \
    AEGIS_MODEL_DIR=/app/models/deberta_v3_small_aegis_v1

RUN groupadd --gid 10001 aegis \
    && useradd --uid 10001 --gid aegis --shell /usr/sbin/nologin --create-home aegis \
    && mkdir -p /app/models /app/logs \
    && chown -R aegis:aegis /app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

USER aegis
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"

ENTRYPOINT ["aegis-proxy"]
