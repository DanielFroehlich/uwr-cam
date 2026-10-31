"""Configuration models and YAML loader for uwr-cam-engine."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ConfigurationError(ValueError):
    """Raised when configuration validation or environment verification fails."""


class OBSConfig(BaseModel):
    """OBS Studio WebSocket v5 connection settings."""
    host: str = Field(default="localhost", description="OBS WebSocket host IP or hostname")
    port: int = Field(default=4455, ge=1, le=65535, description="OBS WebSocket port (default: 4455)")
    password: str = Field(default="", description="OBS WebSocket password (empty if auth disabled)")
    timeout_sec: float = Field(default=3.0, gt=0.0, description="Connection and request timeout in seconds")
    reconnect_interval_sec: float = Field(default=3.0, gt=0.0, description="Auto-reconnect interval in seconds")


class CameraWeights(BaseModel):
    """Weights for composite score calculation."""
    ball: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for ball detection confidence (0.0 to 1.0)")
    motion: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for motion density intensity (0.0 to 1.0)")

    @model_validator(mode="after")
    def validate_weights_sum(self) -> CameraWeights:
        total = self.ball + self.motion
        if total <= 0:
            raise ValueError("Sum of ball and motion weights must be strictly greater than 0")
        return self


class CameraConfig(BaseModel):
    """Configuration for an ingested camera feed and its OBS mapping."""
    id: str = Field(..., min_length=1, max_length=64, description="Unique camera identifier, e.g. cam1")
    name: str = Field(..., description="Human-readable camera description")
    rtsp_url: str = Field(..., description="RTSP URL for low-res computer vision stream")
    obs_scene: str = Field(..., min_length=1, description="Mapped OBS Studio program scene name")
    weights: Optional[CameraWeights] = Field(default=None, description="Scoring weights")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError(f"Camera ID '{v}' may only contain alphanumeric characters, hyphens, and underscores")
        return v

    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("rtsp://"):
            raise ValueError(f"Invalid RTSP URL '{v}': must start with 'rtsp://'")
        return v

    @field_validator("obs_scene")
    @classmethod
    def validate_scene(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("obs_scene cannot be empty")
        return v


class SwitchingConfig(BaseModel):
    """Arbitration and switching decision parameters."""
    fallback_scene: str = Field(default="base", min_length=1, description="Default fallback scene")
    min_dwell_time_sec: float = Field(
        default=2.5,
        ge=0.5,
        le=30.0,
        description="Minimum seconds to stay on a camera before switching (anti-flapping)"
    )
    hysteresis_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score margin required to switch away from current camera"
    )
    fallback_timeout_sec: float = Field(
        default=5.0,
        ge=1.0,
        description="Seconds without detections before reverting to fallback scene"
    )
    manual_override: bool = Field(
        default=False,
        description="Pause automated switching to allow director manual control"
    )
    scene_poll_interval_sec: float = Field(
        default=0.5,
        ge=0.1,
        le=10.0,
        description="Interval in seconds to poll OBS for active program scene changes"
    )
    default_weights: CameraWeights = Field(
        default_factory=CameraWeights,
        description="Default scoring weights for cameras without dedicated weights"
    )


class MarkerConfig(BaseModel):
    """Configuration for real-time bounding box ball marker overlay in OBS."""
    enabled: bool = Field(default=False, description="Enable ball marker bounding box overlay")
    source_name: str = Field(default="Ball_Marker", min_length=1, description="Name of the OBS marker image source")
    update_interval_sec: float = Field(
        default=0.1,
        ge=0.033,
        le=5.0,
        description="Interval in seconds between marker transform updates"
    )
    border_color: List[int] = Field(default=[0, 255, 0], description="RGB border color")
    border_thickness: int = Field(default=4, ge=1, le=32, description="Marker border thickness in pixels")


class OverlayConfig(BaseModel):
    """Configuration for live on-screen text overlay in OBS."""
    enabled: bool = Field(default=False, description="Enable real-time technical metrics overlay in OBS")
    source_name: str = Field(default="CV_Scores_Overlay", min_length=1, description="Name of the text source in OBS")
    update_interval_sec: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Interval in seconds between overlay text updates"
    )
    font_face: str = Field(default="Helvetica", description="Font family name")
    font_size: int = Field(default=36, ge=8, le=256, description="Font point size")
    font_style: str = Field(default="Bold", description="Font style (e.g. Regular, Bold)")
    outline: bool = Field(default=True, description="Enable text outline for visibility")
    drop_shadow: bool = Field(default=True, description="Enable drop shadow for visibility")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Console log level (DEBUG, INFO, WARNING, ERROR)")
    status_interval_sec: float = Field(
        default=5.0,
        ge=0.5,
        description="Interval in seconds between periodic status summaries at INFO level"
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        v = v.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"Invalid log level '{v}'. Allowed levels: {sorted(allowed)}")
        return v


class BallDetectorConfig(BaseModel):
    """Configuration for deep-learning ball detection."""
    enabled: bool = Field(default=True, description="Enable YOLO ball detection")
    model_name: str = Field(
        default="yolov8s-worldv2.pt",
        description="YOLO model path or identifier (e.g. yolov8s-worldv2.pt or custom .pt)",
    )
    confidence_threshold: float = Field(
        default=0.25,
        ge=0.05,
        le=1.0,
        description="Minimum detection confidence threshold",
    )
    classes: List[str] = Field(
        default=["underwater ball", "white ball"],
        description="Open-vocabulary prompts for YOLO-World detection",
    )


class EngineConfig(BaseModel):
    """Root configuration model for uwr-cam-engine."""
    version: str = Field(default="1.0", description="Configuration schema version")
    obs: OBSConfig = Field(default_factory=OBSConfig, description="OBS WebSocket connection settings")
    switching: SwitchingConfig = Field(default_factory=SwitchingConfig, description="Switching arbitration settings")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging settings")
    ball_detector: BallDetectorConfig = Field(
        default_factory=BallDetectorConfig,
        description="YOLO ball detector settings",
    )
    overlay: OverlayConfig = Field(
        default_factory=OverlayConfig,
        description="OBS text overlay settings",
    )
    marker: MarkerConfig = Field(
        default_factory=MarkerConfig,
        description="OBS ball marker overlay settings",
    )
    cameras: List[CameraConfig] = Field(
        ...,
        min_length=2,
        max_length=6,
        description="List of 2 to 6 camera configurations"
    )

    @model_validator(mode="after")
    def validate_global_invariants(self) -> EngineConfig:
        # 0. Populate default weights for cameras without dedicated weights
        for c in self.cameras:
            if c.weights is None:
                c.weights = self.switching.default_weights.model_copy()

        # 1. Unique camera IDs
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            duplicates = {cid for cid in ids if ids.count(cid) > 1}
            raise ValueError(f"Duplicate camera IDs found: {duplicates}")

        # 2. Unique RTSP URLs
        urls = [c.rtsp_url for c in self.cameras]
        if len(urls) != len(set(urls)):
            dup_urls = {u for u in urls if urls.count(u) > 1}
            raise ValueError(f"Duplicate RTSP URLs found: {dup_urls}")

        # 3. Unique OBS Scene mappings
        scenes = [c.obs_scene for c in self.cameras]
        if len(scenes) != len(set(scenes)):
            dup_scenes = {s for s in scenes if scenes.count(s) > 1}
            raise ValueError(f"Duplicate OBS scene mappings found: {dup_scenes}")

        # 4. Fallback scene should not conflict with an individual camera scene
        # (fallback is an overview / master scene)
        if self.switching.fallback_scene in scenes:
            raise ValueError(
                f"Fallback scene '{self.switching.fallback_scene}' cannot be mapped to an individual camera feed"
            )

        return self


def load_config(config_path: str | Path) -> EngineConfig:
    """Load, parse, and validate uwr-cam-engine configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML content in {path}: expected a dictionary root")

    return EngineConfig.model_validate(raw_data)
