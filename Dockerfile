# ─────────────────────────────────────────────────────────────────
# QuantBengal Pro SMC Terminal — Dockerfile
# Multi-stage build: slim final image (~250 MB vs ~900 MB full)
#
# Build : docker build -t quantbengal-smc .
# Run   : docker run -p 8501:8501 quantbengal-smc
# Access: http://localhost:8501
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────
FROM python:3.11-slim AS builder

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy only the dependency manifest first — Docker layer cache means
# this layer is only rebuilt when requirements.txt changes, not on
# every source code edit.
COPY requirements.txt .

# Install into an isolated prefix so we can copy cleanly to final stage
RUN pip install --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit reads these env vars at startup — avoids needing
    # a separate config file for basic headless settings
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false

# Non-root user — security best practice for containerised apps
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source and Streamlit config
COPY app.py .
COPY requirements.txt .
COPY .streamlit/ .streamlit/

# Transfer ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port Streamlit listens on
EXPOSE 8501

# Health-check: Railway / Render / ECS use this to confirm the
# container is healthy before routing traffic to it
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Entrypoint — runs the SMC terminal directly (no shell wrapper needed)
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]
