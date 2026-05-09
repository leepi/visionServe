"""Integration tests for the FastAPI app."""
from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client_with_model(tiny_checkpoint: Path, monkeypatch):
    """Build a TestClient with the model loaded."""
    monkeypatch.setenv("VISIONSERVE_CHECKPOINT_PATH", str(tiny_checkpoint))
    monkeypatch.setenv("VISIONSERVE_DEVICE", "cpu")

    # Force a fresh import so lifespan picks up new env vars
    import importlib

    import visionserve.api.main as main_module
    import visionserve.api.settings as settings_module
    importlib.reload(settings_module)
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client


@pytest.fixture
def client_no_model(monkeypatch, tmp_path):
    """Build a TestClient without a valid checkpoint."""
    monkeypatch.setenv("VISIONSERVE_CHECKPOINT_PATH", str(tmp_path / "nope.pt"))
    monkeypatch.setenv("VISIONSERVE_DEVICE", "cpu")

    import importlib

    import visionserve.api.main as main_module
    import visionserve.api.settings as settings_module
    importlib.reload(settings_module)
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client


def _make_image_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    img = Image.new("RGB", size, color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.integration
def test_root_endpoint(client_with_model):
    r = client_with_model.get("/")
    assert r.status_code == 200
    assert r.json()["name"] == "VisionServe"


@pytest.mark.integration
def test_health_with_model(client_with_model):
    r = client_with_model.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["classes"] == 10


@pytest.mark.integration
def test_health_without_model(client_no_model):
    r = client_no_model.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["model_loaded"] is False


@pytest.mark.integration
def test_predict_single(client_with_model):
    img = _make_image_bytes()
    r = client_with_model.post(
        "/predict", files={"file": ("test.png", img, "image/png")}
    )
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 5  # default top_k
    assert body["model_backbone"] == "resnet18"
    assert body["inference_time_ms"] > 0


@pytest.mark.integration
def test_predict_topk_param(client_with_model):
    img = _make_image_bytes()
    r = client_with_model.post(
        "/predict?top_k=3", files={"file": ("test.png", img, "image/png")}
    )
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 3


@pytest.mark.integration
def test_predict_invalid_topk(client_with_model):
    img = _make_image_bytes()
    r = client_with_model.post(
        "/predict?top_k=0", files={"file": ("test.png", img, "image/png")}
    )
    assert r.status_code == 400


@pytest.mark.integration
def test_predict_empty_file(client_with_model):
    r = client_with_model.post(
        "/predict", files={"file": ("empty.png", b"", "image/png")}
    )
    assert r.status_code == 400


@pytest.mark.integration
def test_predict_batch(client_with_model):
    img = _make_image_bytes()
    files = [("files", (f"img{i}.png", img, "image/png")) for i in range(3)]
    r = client_with_model.post("/predict/batch", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["results"]) == 3


@pytest.mark.integration
def test_predict_no_model(client_no_model):
    img = _make_image_bytes()
    r = client_no_model.post(
        "/predict", files={"file": ("test.png", img, "image/png")}
    )
    assert r.status_code == 503


@pytest.mark.integration
def test_metrics_endpoint(client_with_model):
    # Hit predict to generate some metrics
    img = _make_image_bytes()
    client_with_model.post("/predict", files={"file": ("t.png", img, "image/png")})
    r = client_with_model.get("/metrics")
    assert r.status_code == 200
    assert "visionserve_requests_total" in r.text
