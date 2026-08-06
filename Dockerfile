# ---------------------------------------------------------------------------
# ALM — Agent Language Model
#
# Slim runtime image. Optional extras (local inference, training, pgvector) are
# deliberately not baked in: an image carrying torch for a deployment that talks
# to vLLM over HTTP is several gigabytes of nothing.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALM_HOME=/data/.alm \
    ALM_PACKS_DIR=/app/packs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first, so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY alm ./alm
RUN pip install --no-cache-dir .

COPY packs ./packs

RUN useradd --create-home --uid 10001 alm \
    && mkdir -p /data \
    && chown -R alm:alm /data /app
USER alm

VOLUME ["/data"]
EXPOSE 8800

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8800/v1/health || exit 1

ENTRYPOINT ["alm"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8800"]
