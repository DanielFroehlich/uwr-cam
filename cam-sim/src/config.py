"""Configuration models and loader for cam-sim."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ServerConfig(BaseModel):
    """RTSP server settings."""
    rtsp_port: int = Field(default=8554, ge=1, le=65535, description="Port for RTSP server")
    bind_address: str = Field(default="0.0.0.0", description="IP address to bind the RTSP server to")
    rtsp_transports: List[str] = Field(
        default_factory=lambda: ["tcp", "udp"],
        description="Allowed RTSP media transports: ['tcp', 'udp'], ['tcp'], or ['udp']"
    )

    @field_validator("rtsp_transports")
    @classmethod
    def validate_transports(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("rtsp_transports cannot be empty")
        valid = {"tcp", "udp"}
        cleaned = [t.strip().lower() for t in v]
        invalid = [t for t in cleaned if t not in valid]
        if invalid:
            raise ValueError(f"Invalid RTSP transport(s): {invalid}. Allowed values: 'tcp', 'udp'")
        return cleaned


class DefaultsConfig(BaseModel):
    """Default video encoding options."""
    loop: bool = Field(default=True, description="Loop input video indefinitely")
    video_codec: str = Field(default="libx264", description="FFmpeg video encoder")
    preset: str = Field(default="ultrafast", description="x264 preset")
    tune: str = Field(default="zerolatency", description="x264 tune setting")
    gop_size: int = Field(default=15, ge=1, description="Keyframe distance in frames (15 frames = 0.5s at 30fps)")


class FeedConfig(BaseModel):
    """Configuration for an individual video stream feed."""
    enabled: bool = Field(default=True, description="Whether this feed is active")
    path: str = Field(..., description="RTSP URL path, e.g. /cam1/low")
    width: int = Field(..., gt=0, description="Target frame width in pixels")
    height: int = Field(..., gt=0, description="Target frame height in pixels")
    fps: int = Field(default=30, gt=0, description="Output frames per second")
    bitrate: str = Field(default="1000k", description="Output video bitrate (e.g. 1000k, 6M)")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("/"):
            v = "/" + v
        if len(v) < 2:
            raise ValueError(f"RTSP stream path '{v}' must be at least 2 characters long")
        return v

    @field_validator("width", "height")
    @classmethod
    def validate_even_dimensions(cls, v: int, info) -> int:
        if v % 2 != 0:
            raise ValueError(f"{info.field_name} must be an even number for H.264 encoding (got {v})")
        return v

    @field_validator("bitrate")
    @classmethod
    def validate_bitrate(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^\d+[kKmMgG]?$", v):
            raise ValueError(f"Bitrate '{v}' is invalid. Expected format like '1000k' or '6M'")
        return v


class CameraFeedsConfig(BaseModel):
    """Dual-feed configuration for a camera."""
    low_res: FeedConfig
    high_res: FeedConfig


class CameraConfig(BaseModel):
    """Configuration for a single simulated camera."""
    id: str = Field(..., min_length=1, max_length=64, description="Unique camera identifier")
    name: str = Field(..., description="Human-readable camera name")
    source: str = Field(..., description="Path to source MP4 video file")
    feeds: CameraFeedsConfig

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError(f"Camera ID '{v}' may only contain alphanumeric characters, hyphens, and underscores")
        return v


class CamSimConfig(BaseModel):
    """Root configuration for cam-sim."""
    version: str = Field(default="1.0")
    server: ServerConfig = Field(default_factory=ServerConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    cameras: List[CameraConfig] = Field(..., min_length=2, max_length=6)

    @model_validator(mode="after")
    def validate_global_rules(self) -> CamSimConfig:
        # Check unique camera IDs
        camera_ids = [c.id for c in self.cameras]
        if len(camera_ids) != len(set(camera_ids)):
            duplicates = [cid for cid in camera_ids if camera_ids.count(cid) > 1]
            raise ValueError(f"Duplicate camera IDs found: {set(duplicates)}")

        # Check unique stream paths across all active feeds
        paths: List[str] = []
        for c in self.cameras:
            if c.feeds.low_res.enabled:
                paths.append(c.feeds.low_res.path)
            if c.feeds.high_res.enabled:
                paths.append(c.feeds.high_res.path)

        if len(paths) != len(set(paths)):
            dup_paths = [p for p in paths if paths.count(p) > 1]
            raise ValueError(f"Duplicate RTSP stream paths found across cameras: {set(dup_paths)}")

        return self


def load_config(config_path: str | Path) -> CamSimConfig:
    """Load and validate cam-sim configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML content in {path}: expected a dictionary root")

    return CamSimConfig.model_validate(raw_data)
