# Serving image for the uplift model. Multi-stage so the runtime layer carries
# neither build tools nor the 170 MB parquet used for training.
FROM python:3.12-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp is LightGBM's OpenMP runtime and is required at inference time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=build /install /usr/local

WORKDIR /app
COPY src/ /app/src/
COPY models/current/ /app/models/current/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    UPLIFT_MODEL_DIR=/app/models/current

USER app
EXPOSE 8000

# Hits the real readiness check: /health reports model_loaded, not just liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"model_loaded":true' || exit 1

CMD ["uvicorn", "uplift.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
