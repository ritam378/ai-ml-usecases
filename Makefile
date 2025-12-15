.PHONY: help install install-dev test test-coverage lint format clean docs

# Default target
help:
	@echo "AI Use Cases - Development Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install core dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make install-all      Install all dependencies (dev, llm, cv, nlp)"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run all linters (flake8, mypy, pylint)"
	@echo "  make format           Format code with black and isort"
	@echo "  make type-check       Run mypy type checking"
	@echo "  make pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts and cache"
	@echo "  make clean-test       Remove test artifacts"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build documentation"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-all:
	pip install -e ".[all]"
	pre-commit install

# Testing targets
test:
	pytest

test-coverage:
	pytest --cov=common --cov=08_generative-ai-llms/01_text-to-sql/src \
		--cov-report=html --cov-report=term-missing

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-fast:
	pytest -m "not slow"

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 common/ */*/src/ --max-line-length=100 --extend-ignore=E203,E501,W503
	@echo "Running mypy..."
	mypy common/ --ignore-missing-imports
	@echo "Running pylint..."
	pylint common/ --max-line-length=100 || true

format:
	@echo "Running isort..."
	isort common/ */*/src/ --profile=black --line-length=100
	@echo "Running black..."
	black common/ */*/src/ --line-length=100

type-check:
	mypy common/ */*/src/ --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

# Cleanup targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ htmlcov/ .coverage

clean-test:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -f .coverage

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && make html

# Database targets (for text-to-sql)
create-db:
	python 08_generative-ai-llms/01_text-to-sql/data/create_database.py

# Quick validation
check: format lint test-fast
	@echo "✓ All checks passed!"
