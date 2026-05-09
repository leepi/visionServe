# VisionServe

[![CI](https://github.com/YOUR_USERNAME/visionserve/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/visionserve/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)

Production-ready image classification service. Train a model, serve it through a FastAPI endpoint, monitor with Prometheus, ship with Docker.

## Highlights

- **Two backbones** — ResNet18 and MobileNetV3-Small, swappable via YAML config
- **Two dataset modes** — CIFAR-10 (auto-download) and any ImageFolder layout
- **REST API** — single + batch prediction, health checks, Prometheus metrics
- **Production hygiene** — multi-stage Dockerfile, non-root user, healthchecks, GitHub Actions CI, GHCR publishing
- **Tested** — 33 tests covering config, model factory, transforms, inference engine, and full API integration
- **Reproducible** — seeded training, config-driven runs, checkpoint metadata

## Quickstart

### Run the API with the included starter checkpoint

```bash
git clone https://github.com/YOUR_USERNAME/visionserve.git
cd visionserve
pip install -e .
visionserve-api
# In another terminal:
curl http://localhost:8000/healthz
curl -X POST -F "file=@your_image.jpg" "http://localhost:8000/predict?top_k=5"
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Or with Docker

```bash
docker compose up --build
```

### Train your own model on CIFAR-10

```bash
pip install -e ".[train]"
visionserve-train --config configs/cifar10_resnet18.yaml
visionserve-eval --config configs/cifar10_resnet18.yaml --checkpoint checkpoints/best.pt
```

## Project layout

```
visionserve/
├── src/visionserve/
│   ├── api/              # FastAPI app, schemas, settings
│   ├── data/             # Datasets and transforms
│   ├── models/           # Model factory (resnet18 / mobilenet_v3_small)
│   ├── utils/            # Seed, device, logging
│   ├── config.py         # YAML config dataclasses
│   ├── inference.py      # Inference engine wrapping a checkpoint
│   ├── train.py          # Training entry point
│   └── evaluate.py       # Evaluation entry point
├── configs/              # YAML configs for different setups
├── tests/                # pytest suite
├── checkpoints/          # Saved models (best.pt is the starter)
├── scripts/              # Utility scripts
├── Dockerfile            # Multi-stage CPU build
├── docker-compose.yml
└── .github/workflows/    # CI/CD
```

## API reference

| Method | Path             | Description                                     |
|--------|------------------|-------------------------------------------------|
| GET    | `/`              | Service metadata                                |
| GET    | `/healthz`       | Liveness + readiness (model load status)        |
| GET    | `/metrics`       | Prometheus metrics                              |
| GET    | `/docs`          | Swagger UI                                      |
| POST   | `/predict`       | Classify a single uploaded image (form field: `file`) |
| POST   | `/predict/batch` | Classify a batch of images (form field: `files`)|

Both prediction endpoints accept a `top_k` query parameter (default 5).

### Example response

```json
{
  "predictions": [
    {"class": "ship", "probability": 0.9999},
    {"class": "automobile", "probability": 0.00004},
    {"class": "cat", "probability": 0.00002}
  ],
  "model_backbone": "resnet18",
  "inference_time_ms": 29.05
}
```

## Configuration

Training is driven by YAML configs. A few defaults are provided:

- `configs/cifar10_resnet18.yaml` — CIFAR-10 with ResNet18 backbone
- `configs/cifar10_mobilenet.yaml` — CIFAR-10 with MobileNetV3-Small
- `configs/custom_imagefolder.yaml` — Template for your own ImageFolder dataset

CLI flags override config values:

```bash
visionserve-train --config configs/cifar10_resnet18.yaml --epochs 20 --backbone mobilenet_v3_small
```

The API reads its settings from environment variables (prefix `VISIONSERVE_`):

| Variable                          | Default                    |
|-----------------------------------|----------------------------|
| `VISIONSERVE_CHECKPOINT_PATH`     | `./checkpoints/best.pt`    |
| `VISIONSERVE_DEVICE`              | `auto` (cuda → mps → cpu)  |
| `VISIONSERVE_MAX_BATCH_SIZE`      | `32`                       |
| `VISIONSERVE_MAX_IMAGE_SIZE_MB`   | `10`                       |
| `VISIONSERVE_HOST` / `_PORT`      | `0.0.0.0` / `8000`         |

## Custom dataset

Organize your data in ImageFolder layout:

```
data/
├── train/
│   ├── class_a/img1.jpg img2.jpg ...
│   └── class_b/...
└── val/
    ├── class_a/...
    └── class_b/...
```

Edit `configs/custom_imagefolder.yaml` to point to those paths and run `visionserve-train --config configs/custom_imagefolder.yaml`. Class names are inferred from directory names.

## Development

```bash
pip install -e ".[dev,train]"
pre-commit install
make test    # pytest with coverage
make lint    # ruff + mypy
make format  # ruff format + autofix
```

## CI/CD

GitHub Actions runs lint → tests (Python 3.10 and 3.11) → Docker build on every push. Tagging a release (e.g., `v0.1.0`) publishes the image to GitHub Container Registry at `ghcr.io/YOUR_USERNAME/visionserve`.

## About the starter checkpoint

The `checkpoints/best.pt` shipped in this repo is a real trained ResNet18 — but it was trained on a small synthetic colored-shapes dataset (the CIFAR-10 download host wasn't reachable from the build environment). It demonstrates the full pipeline end-to-end and reaches ~89% validation accuracy on the synthetic data. **For real CIFAR-10 performance, run `make train` locally** — the dataset will download cleanly and you'll get the proper model. The checkpoint format, API contract, and class labels match either way.

## License

[MIT](LICENSE)
