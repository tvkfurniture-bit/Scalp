# ─────────────────────────────────────────────────────────────────
# QuantBengal Pro SMC Terminal — Dockerfile
# Multi-stage build: slim final image (~250 MB)
#
# Build : docker build -t quantbengal-smc .
# Run   : docker run -p 8501:8501 quantbengal-smc
# Access: http://localhost:8501
#
# Note: config.toml is set for local dev (headless=false, CORS=true).
# The ENV block below overrides those values for the container so
# Docker/cloud always gets the correct headless server behaviour.
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────
FROM python:3.11-slim AS runtime

# ── Streamlit env overrides ───────────────────────────────────────
# These override config.toml values at runtime.
# config.toml has headless=false for local dev;
# these env vars flip to headless server mode inside Docker/cloud.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Non-root user — security best practice
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application files
COPY app.py .
COPY requirements.txt .
COPY .streamlit/ .streamlit/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# Health-check used by Railway / Render / ECS to gate traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
