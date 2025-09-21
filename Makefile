# Smart City Computer Vision - Development Makefile

.PHONY: help install install-dev clean test lint format check-format type-check security-check docker-build docker-run train-all evaluate-all setup-env

# Default target
help: ## Show this help message
	@echo "Smart City Computer Vision - Development Commands"
	@echo "================================================"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation targets
install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 mypy bandit safety pre-commit

# Environment setup
setup-env: ## Set up complete development environment
	python -m venv venv
	.\venv\Scripts\activate && pip install --upgrade pip
	.\venv\Scripts\activate && $(MAKE) install-dev
	.\venv\Scripts\activate && pre-commit install
	python setup.py

# Code quality targets
clean: ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf dist/
	rm -rf build/

format: ## Format code with black and isort
	black .
	isort .

check-format: ## Check code formatting
	black --check --diff .
	isort --check-only --diff .

lint: ## Run linting with flake8
	flake8 .

type-check: ## Run type checking with mypy
	mypy . --ignore-missing-imports

security-check: ## Run security checks
	bandit -r . -x tests/
	safety check

# Testing targets
test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest -m "unit"

test-integration: ## Run integration tests only
	pytest -m "integration"

test-slow: ## Run slow tests
	pytest -m "slow"

test-fast: ## Run fast tests only
	pytest -m "not slow"

test-coverage: ## Run tests with coverage report
	pytest --cov=. --cov-report=html --cov-report=term-missing

# Training targets
train-garbage: ## Train garbage detection model
	python garbage_train.py

train-helmet: ## Train helmet detection model
	python helmet_train.py

train-traffic: ## Train traffic detection model
	python traffic_train.py

train-all: ## Train all models sequentially
	@echo "Training all Smart City Computer Vision models..."
	$(MAKE) train-garbage
	$(MAKE) train-helmet
	$(MAKE) train-traffic

# Evaluation targets
evaluate-garbage: ## Evaluate garbage detection model
	python evaluate.py --model garbage --benchmark --compare

evaluate-helmet: ## Evaluate helmet detection model
	python evaluate.py --model helmet --benchmark --compare

evaluate-traffic: ## Evaluate traffic detection model
	python evaluate.py --model traffic --benchmark --compare

evaluate-all: ## Evaluate all models
	python evaluate.py --model all --benchmark --compare

# Dataset utilities
validate-datasets: ## Validate all datasets
	python dataset_utils.py --action validate --model garbage
	python dataset_utils.py --action validate --model helmet
	python dataset_utils.py --action validate --model traffic

analyze-datasets: ## Analyze all datasets
	python dataset_utils.py --action analyze --model garbage --output analysis_garbage
	python dataset_utils.py --action analyze --model helmet --output analysis_helmet
	python dataset_utils.py --action analyze --model traffic --output analysis_traffic

# Demo targets
demo-garbage: ## Run garbage detection demo with webcam
	python demo.py --model garbage --source 0

demo-helmet: ## Run helmet detection demo with webcam
	python demo.py --model helmet --source 0

demo-traffic: ## Run traffic detection demo with webcam
	python demo.py --model traffic --source 0

# Docker targets
docker-build: ## Build Docker image
	docker build -t smartcity-cv:latest .

docker-run: ## Run Docker container
	docker run --rm -it smartcity-cv:latest

docker-compose-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

docker-compose-logs: ## View logs from all services
	docker-compose logs -f

# Development targets
pre-commit: ## Run pre-commit hooks manually
	pre-commit run --all-files

check-all: ## Run all quality checks
	$(MAKE) check-format
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) security-check
	$(MAKE) test

# Documentation targets
docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

# Release targets
version-patch: ## Bump patch version
	@echo "Version bumping not implemented yet"

version-minor: ## Bump minor version
	@echo "Version bumping not implemented yet"

version-major: ## Bump major version
	@echo "Version bumping not implemented yet"

# Monitoring targets
monitor: ## Start monitoring stack
	docker-compose --profile monitoring up -d

monitor-down: ## Stop monitoring stack
	docker-compose --profile monitoring down

# Quick development shortcuts
dev-setup: setup-env ## Alias for setup-env

quick-test: test-fast ## Alias for test-fast

full-check: check-all ## Alias for check-all

all-models: train-all evaluate-all ## Train and evaluate all models