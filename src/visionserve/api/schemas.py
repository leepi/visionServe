"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    class_: str = Field(alias="class")
    probability: float = Field(ge=0.0, le=1.0)

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model_backbone: str
    inference_time_ms: float


class BatchPredictResponse(BaseModel):
    results: list[list[Prediction]]
    count: int
    model_backbone: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backbone: str | None = None
    classes: int | None = None
    device: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
