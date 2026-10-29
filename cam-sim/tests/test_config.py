"""Unit tests for configuration validation in cam-sim."""

import pytest
from pydantic import ValidationError

from src.config import (
    CameraConfig,
    CameraFeedsConfig,
    CamSimConfig,
    DefaultsConfig,
    FeedConfig,
    ServerConfig,
    load_config,
)


def sample_camera(cam_id: str, low_path: str, high_path: str) -> dict:
    return {
        "id": cam_id,
        "name": f"Camera {cam_id}",
        "source": f"/videos/{cam_id}.mp4",
        "feeds": {
            "low_res": {
                "enabled": True,
                "path": low_path,
                "width": 640,
                "height": 360,
                "fps": 30,
                "bitrate": "1000k",
            },
            "high_res": {
                "enabled": True,
                "path": high_path,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "bitrate": "6000k",
            },
        },
    }


def test_valid_config(tmp_path):
    """Verify that a properly structured config file parses without error."""
    yaml_content = """
version: "1.0"
server:
  rtsp_port: 8554
  bind_address: "0.0.0.0"
defaults:
  loop: true
  video_codec: "libx264"
  preset: "ultrafast"
  tune: "zerolatency"
  gop_size: 30
cameras:
  - id: "cam1"
    name: "Camera 1"
    source: "/videos/cam1.mp4"
    feeds:
      low_res:
        enabled: true
        path: "/cam1/low"
        width: 640
        height: 360
        fps: 30
        bitrate: "1000k"
      high_res:
        enabled: true
        path: "/cam1/high"
        width: 1920
        height: 1080
        fps: 30
        bitrate: "6000k"
  - id: "cam2"
    name: "Camera 2"
    source: "/videos/cam2.mp4"
    feeds:
      low_res:
        enabled: true
        path: "/cam2/low"
        width: 640
        height: 360
        fps: 30
        bitrate: "1000k"
      high_res:
        enabled: true
        path: "/cam2/high"
        width: 1920
        height: 1080
        fps: 30
        bitrate: "6000k"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(config_file)
    assert cfg.version == "1.0"
    assert cfg.server.rtsp_port == 8554
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].id == "cam1"
    assert cfg.cameras[1].id == "cam2"


def test_camera_count_limits():
    """Verify that minimum 2 and maximum 6 cameras are strictly enforced."""
    raw = {
        "version": "1.0",
        "cameras": [
            sample_camera("cam1", "/cam1/low", "/cam1/high")
        ]
    }
    # 1 camera should fail
    with pytest.raises(ValidationError) as exc:
        CamSimConfig.model_validate(raw)
    assert "at least 2 items" in str(exc.value)

    # 7 cameras should fail
    raw["cameras"] = [
        sample_camera(f"cam{i}", f"/cam{i}/low", f"/cam{i}/high")
        for i in range(1, 8)
    ]
    with pytest.raises(ValidationError) as exc:
        CamSimConfig.model_validate(raw)
    assert "at most 6 items" in str(exc.value)


def test_duplicate_camera_id():
    """Verify that duplicate camera IDs are rejected."""
    raw = {
        "version": "1.0",
        "cameras": [
            sample_camera("cam1", "/cam1/low", "/cam1/high"),
            sample_camera("cam1", "/cam2/low", "/cam2/high"),
        ]
    }
    with pytest.raises(ValidationError) as exc:
        CamSimConfig.model_validate(raw)
    assert "Duplicate camera IDs found" in str(exc.value)


def test_duplicate_stream_path():
    """Verify that duplicate RTSP paths across feeds are rejected."""
    raw = {
        "version": "1.0",
        "cameras": [
            sample_camera("cam1", "/shared/path", "/cam1/high"),
            sample_camera("cam2", "/shared/path", "/cam2/high"),
        ]
    }
    with pytest.raises(ValidationError) as exc:
        CamSimConfig.model_validate(raw)
    assert "Duplicate RTSP stream paths found" in str(exc.value)


def test_odd_dimensions_rejected():
    """Verify that odd pixel dimensions are rejected because H.264 requires even dimensions."""
    feed = {
        "enabled": True,
        "path": "/test",
        "width": 641,  # Odd width
        "height": 360,
        "fps": 30,
        "bitrate": "1000k",
    }
    with pytest.raises(ValidationError) as exc:
        FeedConfig.model_validate(feed)
    assert "even number" in str(exc.value)


def test_invalid_bitrate_rejected():
    """Verify that malformed bitrate strings are rejected."""
    feed = {
        "enabled": True,
        "path": "/test",
        "width": 640,
        "height": 360,
        "fps": 30,
        "bitrate": "fast",
    }
    with pytest.raises(ValidationError) as exc:
        FeedConfig.model_validate(feed)
    assert "Bitrate 'fast' is invalid" in str(exc.value)


def test_rtsp_transports_validation():
    """Verify validation of rtsp_transports in ServerConfig."""
    # Valid configurations
    s1 = ServerConfig(rtsp_transports=["tcp"])
    assert s1.rtsp_transports == ["tcp"]

    s2 = ServerConfig(rtsp_transports=["UDP", "TCP"])
    assert set(s2.rtsp_transports) == {"udp", "tcp"}

    # Disallowed transport protocol
    with pytest.raises(ValidationError) as exc:
        ServerConfig(rtsp_transports=["http"])
    assert "Invalid RTSP transport" in str(exc.value)

    # Empty transports list
    with pytest.raises(ValidationError) as exc:
        ServerConfig(rtsp_transports=[])
    assert "rtsp_transports cannot be empty" in str(exc.value)
