.PHONY: install venv test lint fmt typecheck run docker-build docker-up clean

PYTHON ?= python

venv:
	$(PYTHON) -m venv .venv

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests -q --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src cli tests

fmt:
	$(PYTHON) -m ruff check --fix src cli tests

typecheck:
	$(PYTHON) -m mypy src

run:
	$(PYTHON) -m cli.main --help

docker-build:
	docker build -t agentmesh:latest .

docker-up:
	docker compose up --build -d

clean:
	$(PYTHON) -m compileall -q src cli tests
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info htmlcov .coverage
