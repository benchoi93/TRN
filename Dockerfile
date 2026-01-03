# Dockerfile for FastAPI app (uvicorn)
# Built to be small and suitable for Fly.io / Docker-based hosts.

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_VIRTUALENVS_CREATE=false \
    PORT=8080 \
    DB_PATH=/data/survey.db

# Install system deps required for some packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and data volume mountpoint
WORKDIR /app
RUN mkdir -p /app /data

# Copy requirements first for Docker layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt

# Copy the application
COPY . /app

# Make a non-root user and fix permissions. We create the user but keep running as root
# so the process can bind to privileged port 80 inside the container. Note: running as
# root is less secure; consider using setcap or a reverse proxy in production.
RUN useradd --no-log-init --create-home appuser \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8080

# Default command - use uvicorn. Use a shell so $PORT is expanded at runtime by the shell.
# Using the exec form with a literal "${PORT}" doesn't expand the env var and causes errors.
## Use `exec` so the shell forwards signals to uvicorn (graceful shutdowns).
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
