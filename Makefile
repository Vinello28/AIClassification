# AI Classification Makefile
# Use: make <command>

.PHONY: help install install-dev test lint format clean build docker-build docker-run server client train setup

# Default target
help:
	@echo "Available commands:"
	@echo "  help         - Show this help message"
	@echo "  install      - Install package in production mode"
	@echo "  install-dev  - Install package in development mode with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run code linting"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package distribution"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-compose - Run with docker-compose"
	@echo "  server       - Start the API server"
	@echo "  client       - Test the client"
	@echo "  train        - Run model training"
	@echo "  setup        - Initial setup of the environment"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,server]"

# Development
setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  # Linux/Mac"
	@echo "  venv\\Scripts\\activate     # Windows"
	@echo "Then run: make install-dev"

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src/ai_classification --cov-report=html

# Code quality
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build
build: clean
	python setup.py sdist bdist_wheel

# Docker
docker-build:
	docker build -f docker/Dockerfile -t ai-classification:latest .

docker-run:
	docker run -p 8000:8000 --name ai-classification-server ai-classification:latest

docker-compose:
	cd docker && docker-compose up -d

docker-stop:
	cd docker && docker-compose down

# Application
server:
	python server.py

client:
	python client.py

train:
	python -c "from src.ai_classification.core.classifier import AITextClassifier; c=AITextClassifier(); c.train()"

# Model management
download-models:
	@echo "Models will be downloaded automatically on first use"

# Documentation
docs:
	@echo "Documentation generation not yet implemented"

# Environment
env-check:
	python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Complete development setup
dev-setup: setup install-dev env-check
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works"