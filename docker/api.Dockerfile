# API Dockerfile (FastAPI + Uvicorn)
FROM python:3.11-slim AS base

ENV POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

WORKDIR /app

# Copy only dependency files first for better caching
COPY pyproject.toml poetry.lock* ./
RUN poetry install --without dev --no-root --no-ansi

# Copy source
COPY src ./src

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 8080
CMD ["poetry", "run", "uvicorn", "ai_interviewer_pm.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--loop", "asyncio"]

