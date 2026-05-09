"""FastAPI application."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_client.registry import REGISTRY

from visionserve import __version__
from visionserve.api.schemas import (
    BatchPredictResponse,
    HealthResponse,
    PredictResponse,
    Prediction,
)
from visionserve.api.settings import get_settings
from visionserve.inference import InferenceEngine
from visionserve.utils import get_logger

logger = get_logger("visionserve.api")

# Prometheus metrics — registered once, idempotent across module reloads
def _get_or_create_metric(metric_cls, name: str, *args, **kwargs):
    existing = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    if existing is not None:
        return existing
    return metric_cls(name, *args, **kwargs)


REQUEST_COUNT = _get_or_create_metric(
    Counter,
    "visionserve_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
INFERENCE_LATENCY = _get_or_create_metric(
    Histogram,
    "visionserve_inference_latency_seconds",
    "Inference latency in seconds",
    ["endpoint"],
)


# Global engine — populated in lifespan
_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global _engine
    settings = get_settings()
    ckpt = Path(settings.checkpoint_path)
    if ckpt.exists():
        try:
            logger.info(f"Loading checkpoint: {ckpt}")
            _engine = InferenceEngine(ckpt, device=settings.device)
            logger.info(
                f"Model ready: backbone={_engine.backbone} "
                f"classes={len(_engine.class_names)} device={_engine.device}"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load checkpoint: {e}")
            _engine = None
    else:
        logger.warning(f"Checkpoint not found at {ckpt}; API will respond unhealthy")

    yield

    _engine = None
    logger.info("API shutdown complete")


app = FastAPI(
    title="VisionServe",
    description="Production-ready image classification API",
    version=__version__,
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check checkpoint path.",
        )
    return _engine


async def _read_image_bytes(file: UploadFile, max_mb: int) -> bytes:
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large: {size_mb:.2f} MB (max {max_mb} MB)",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )
    return data


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "name": "VisionServe",
        "version": __version__,
        "endpoints": ["/healthz", "/predict", "/predict/batch", "/metrics", "/docs"],
    }


@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    REQUEST_COUNT.labels(endpoint="/healthz", status="200").inc()
    if _engine is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        backbone=_engine.backbone,
        classes=len(_engine.class_names),
        device=str(_engine.device),
    )


@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"description": "Model not loaded"}},
)
async def predict(
    file: UploadFile = File(..., description="Image file (jpg/png/etc.)"),
    top_k: int = 5,
) -> PredictResponse:
    """Classify a single uploaded image."""
    engine = _require_engine()
    settings = get_settings()
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    try:
        image_bytes = await _read_image_bytes(file, settings.max_image_size_mb)
        t0 = time.perf_counter()
        with INFERENCE_LATENCY.labels(endpoint="/predict").time():
            preds = engine.predict(image_bytes, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except HTTPException:
        REQUEST_COUNT.labels(endpoint="/predict", status="4xx").inc()
        raise
    except Exception as e:  # noqa: BLE001
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
    return PredictResponse(
        predictions=[Prediction(**p) for p in preds],
        model_backbone=engine.backbone,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={503: {"description": "Model not loaded"}},
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple image files"),
    top_k: int = 5,
) -> BatchPredictResponse:
    """Classify a batch of uploaded images."""
    engine = _require_engine()
    settings = get_settings()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} (max {settings.max_batch_size})",
        )
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")

    try:
        images_bytes = [
            await _read_image_bytes(f, settings.max_image_size_mb) for f in files
        ]
        t0 = time.perf_counter()
        with INFERENCE_LATENCY.labels(endpoint="/predict/batch").time():
            results = engine.predict_batch(images_bytes, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except HTTPException:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="4xx").inc()
        raise
    except Exception as e:  # noqa: BLE001
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="500").inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e

    REQUEST_COUNT.labels(endpoint="/predict/batch", status="200").inc()
    return BatchPredictResponse(
        results=[[Prediction(**p) for p in r] for r in results],
        count=len(results),
        model_backbone=engine.backbone,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


def run() -> None:
    """Run the API with uvicorn (used by `visionserve-api` entry point)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "visionserve.api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
