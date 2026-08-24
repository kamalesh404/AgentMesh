FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml setup.py README.md ./
COPY src ./src
COPY cli ./cli
RUN pip install --no-deps -e .

RUN useradd --create-home meshuser && chown -R meshuser /app
USER meshuser

ENTRYPOINT ["agentmesh"]
CMD ["--help"]
