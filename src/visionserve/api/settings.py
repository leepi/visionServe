"""API settings — read from environment."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API settings, configurable via environment variables (prefix VISIONSERVE_)."""

    checkpoint_path: str = "./checkpoints/best.pt"
    device: str = "auto"
    max_batch_size: int = 32
    max_image_size_mb: int = 10
    cors_origins: list[str] = ["*"]
    host: str = "0.0.0.0"  # noqa: S104 — intentional for containerized service
    port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="VISIONSERVE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
