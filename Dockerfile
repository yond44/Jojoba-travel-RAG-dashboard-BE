# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps (build tools only if you have packages that need compiling,
# e.g. some pydantic/uvloop wheels on slim images)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached unless requirements change
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src ./src

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# Basic container-level health check hitting your /health route
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Use uvicorn directly (not python -m src.main) so --workers / --reload
# can be controlled from docker-compose.yml without editing code.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]