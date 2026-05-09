.PHONY: help install install-dev train eval serve test lint format docker-build docker-run clean

help:
	@echo "VisionServe — make targets"
	@echo "  install       Install runtime deps + package"
	@echo "  install-dev   Install dev + train + runtime deps"
	@echo "  train         Train default CIFAR-10 ResNet18 model"
	@echo "  eval          Evaluate trained checkpoint"
	@echo "  serve         Run API locally (uvicorn)"
	@echo "  test          Run pytest with coverage"
	@echo "  lint          Run ruff check + mypy"
	@echo "  format        Run ruff format"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run docker-compose service"
	@echo "  clean         Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,train]"

train:
	visionserve-train --config configs/cifar10_resnet18.yaml

eval:
	visionserve-eval --config configs/cifar10_resnet18.yaml --checkpoint checkpoints/best.pt

serve:
	visionserve-api

test:
	pytest --cov=visionserve --cov-report=term-missing

lint:
	ruff check src tests
	mypy src --ignore-missing-imports || true

format:
	ruff format src tests
	ruff check --fix src tests

docker-build:
	docker build -t visionserve:latest .

docker-run:
	docker compose up --build

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
