.PHONY: install dev test test-cov lint format typecheck train-samples serve clean docker-up docker-down deploy-local deploy-status help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=ml_serving --cov-report=term-missing --cov-report=html

lint: ## Run linter
	ruff check ml_serving/ tests/

format: ## Auto-format code
	ruff format ml_serving/ tests/
	ruff check --fix ml_serving/ tests/

typecheck: ## Run mypy type checker
	mypy ml_serving/

train-samples: ## Train sample models for demos/tests
	python models/sample/train_sample_models.py

serve: ## Start the API server (development)
	uvicorn ml_serving.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up: ## Start all services (API + Prometheus + Grafana)
	docker-compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker-compose -f docker/docker-compose.yml down

deploy-local: ## Deploy locally with Docker
	python scripts/deploy.py --target local --action deploy -y

deploy-status: ## Check local deployment health
	python scripts/deploy.py --target local --action status

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
