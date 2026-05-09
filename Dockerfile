# syntax=docker/dockerfile:1.6

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed to build a wheel
COPY pyproject.toml README.md ./
COPY src ./src

# Install CPU-only PyTorch first (smaller image), then the project
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISIONSERVE_CHECKPOINT_PATH=/app/checkpoints/best.pt \
    VISIONSERVE_HOST=0.0.0.0 \
    VISIONSERVE_PORT=8000

# Non-root user for safety
RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Bring in the checkpoints directory (will contain best.pt at runtime if mounted/copied)
COPY --chown=app:app checkpoints /app/checkpoints

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
import json; \
r = urllib.request.urlopen('http://localhost:8000/healthz', timeout=3); \
sys.exit(0 if r.status == 200 else 1)" || exit 1

CMD ["visionserve-api"]
