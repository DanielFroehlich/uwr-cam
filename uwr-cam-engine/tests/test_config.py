"""Unit tests for uwr-cam-engine configuration models and parser."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from src.config import (
    CameraConfig,
    CameraWeights,
    ConfigurationError,
    EngineConfig,
    LoggingConfig,
    OBSConfig,
    SwitchingConfig,
    load_config,
)


def sample_valid_config_dict() -> dict:
    return {
        "version": "1.0",
        "obs": {
            "host": "127.0.0.1",
            "port": 4455,
            "password": "secret",
            "timeout_sec": 3.0,
            "reconnect_interval_sec": 2.0,
        },
        "switching": {
            "fallback_scene": "base",
            "min_dwell_time_sec": 2.0,
            "hysteresis_threshold": 0.2,
            "fallback_timeout_sec": 4.0,
            "manual_override": False,
        },
        "logging": {"level": "DEBUG"},
        "cameras": [
            {
                "id": "cam1",
                "name": "Goal Left",
                "rtsp_url": "rtsp://localhost:8554/cam1/low",
                "obs_scene": "cam1",
                "weights": {"ball": 0.8, "motion": 0.2},
            },
            {
                "id": "cam2",
                "name": "Goal Right",
                "rtsp_url": "rtsp://localhost:8554/cam2/low",
                "obs_scene": "cam2",
                "weights": {"ball": 0.6, "motion": 0.4},
            },
        ],
    }


def test_valid_config_loading(tmp_path):
    config_file = tmp_path / "config.yaml"
    data = sample_valid_config_dict()
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg.version == "1.0"
    assert cfg.obs.host == "127.0.0.1"
    assert cfg.obs.port == 4455
    assert cfg.obs.password == "secret"
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].id == "cam1"
    assert cfg.cameras[1].id == "cam2"
    assert cfg.switching.fallback_scene == "base"
    assert cfg.switching.min_dwell_time_sec == 2.0
    assert cfg.switching.default_weights.ball == 0.7
    assert cfg.switching.default_weights.motion == 0.3
    assert cfg.cameras[0].weights.ball == 0.8
    assert cfg.cameras[0].weights.motion == 0.2
    assert cfg.overlay.enabled is False
    assert cfg.overlay.source_name == "CV_Scores_Overlay"
    assert cfg.overlay.font_face == "Helvetica"


def test_overlay_config_custom(tmp_path):
    config_file = tmp_path / "config.yaml"
    data = sample_valid_config_dict()
    data["overlay"] = {
        "enabled": True,
        "source_name": "Custom_Overlay",
        "update_interval_sec": 0.5,
        "font_face": "Arial",
        "font_size": 24,
        "font_style": "Regular",
        "outline": False,
        "drop_shadow": False,
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg.overlay.enabled is True
    assert cfg.overlay.source_name == "Custom_Overlay"
    assert cfg.overlay.update_interval_sec == 0.5
    assert cfg.overlay.font_face == "Arial"
    assert cfg.overlay.font_size == 24
    assert cfg.overlay.font_style == "Regular"
    assert cfg.overlay.outline is False
    assert cfg.overlay.drop_shadow is False


def test_marker_config_defaults_and_custom(tmp_path):
    config_file = tmp_path / "config.yaml"
    data = sample_valid_config_dict()
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(config_file)
    assert cfg.marker.enabled is False
    assert cfg.marker.source_name == "Ball_Marker"
    assert cfg.marker.update_interval_sec == 0.1
    assert cfg.marker.border_color == [0, 255, 0]
    assert cfg.marker.border_thickness == 4

    # Custom marker settings
    data["marker"] = {
        "enabled": True,
        "source_name": "Yellow_Box",
        "update_interval_sec": 0.05,
        "border_color": [255, 255, 0],
        "border_thickness": 6,
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg_custom = load_config(config_file)
    assert cfg_custom.marker.enabled is True
    assert cfg_custom.marker.source_name == "Yellow_Box"
    assert cfg_custom.marker.update_interval_sec == 0.05
    assert cfg_custom.marker.border_color == [255, 255, 0]
    assert cfg_custom.marker.border_thickness == 6


def test_missing_config_file():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yaml")


def test_invalid_yaml_syntax(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(": : invalid", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(bad_file)


def test_non_dict_yaml(tmp_path):
    list_file = tmp_path / "list.yaml"
    list_file.write_text("- item1\n- item2", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a dictionary root"):
        load_config(list_file)


def test_min_cameras_constraint():
    data = sample_valid_config_dict()
    data["cameras"] = [data["cameras"][0]]  # Only 1 camera
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(data)


def test_max_cameras_constraint():
    data = sample_valid_config_dict()
    # Create 7 cameras (exceeding limit of 6)
    data["cameras"] = [
        {
            "id": f"cam{i}",
            "name": f"Camera {i}",
            "rtsp_url": f"rtsp://localhost:8554/cam{i}/low",
            "obs_scene": f"cam{i}",
        }
        for i in range(1, 8)
    ]
    with pytest.raises(ValidationError):
        EngineConfig.model_validate(data)


def test_duplicate_camera_ids_rejected():
    data = sample_valid_config_dict()
    data["cameras"][1]["id"] = "cam1"  # Duplicate id
    with pytest.raises(ValidationError, match="Duplicate camera IDs found"):
        EngineConfig.model_validate(data)


def test_duplicate_rtsp_urls_rejected():
    data = sample_valid_config_dict()
    data["cameras"][1]["rtsp_url"] = data["cameras"][0]["rtsp_url"]
    with pytest.raises(ValidationError, match="Duplicate RTSP URLs found"):
        EngineConfig.model_validate(data)


def test_duplicate_obs_scenes_rejected():
    data = sample_valid_config_dict()
    data["cameras"][1]["obs_scene"] = data["cameras"][0]["obs_scene"]
    with pytest.raises(ValidationError, match="Duplicate OBS scene mappings found"):
        EngineConfig.model_validate(data)


def test_fallback_scene_cannot_match_camera_scene():
    data = sample_valid_config_dict()
    data["switching"]["fallback_scene"] = "cam1"
    with pytest.raises(ValidationError, match="Fallback scene 'cam1' cannot be mapped to an individual camera feed"):
        EngineConfig.model_validate(data)


def test_invalid_rtsp_url_format():
    data = sample_valid_config_dict()
    data["cameras"][0]["rtsp_url"] = "http://localhost:8554/cam1"
    with pytest.raises(ValidationError, match="must start with 'rtsp://'"):
        EngineConfig.model_validate(data)


def test_invalid_camera_id_format():
    data = sample_valid_config_dict()
    data["cameras"][0]["id"] = "cam 1 with spaces!"
    with pytest.raises(ValidationError, match="may only contain alphanumeric characters"):
        EngineConfig.model_validate(data)


def test_camera_weights_sum_zero_rejected():
    with pytest.raises(ValidationError, match="strictly greater than 0"):
        CameraWeights(ball=0.0, motion=0.0)


def test_camera_weights_optional_uses_switching_default_weights():
    data = sample_valid_config_dict()
    # Remove weights from cam1 and cam2
    del data["cameras"][0]["weights"]
    del data["cameras"][1]["weights"]
    # Specify custom default_weights in switching
    data["switching"]["default_weights"] = {"ball": 0.4, "motion": 0.6}

    cfg = EngineConfig.model_validate(data)
    assert cfg.switching.default_weights.ball == 0.4
    assert cfg.switching.default_weights.motion == 0.6
    assert cfg.cameras[0].weights is not None
    assert cfg.cameras[0].weights.ball == 0.4
    assert cfg.cameras[0].weights.motion == 0.6
    assert cfg.cameras[1].weights is not None
    assert cfg.cameras[1].weights.ball == 0.4
    assert cfg.cameras[1].weights.motion == 0.6


def test_camera_weights_partial_override():
    data = sample_valid_config_dict()
    # cam1 has no weights -> inherits default_weights
    del data["cameras"][0]["weights"]
    # cam2 has explicit weights
    data["cameras"][1]["weights"] = {"ball": 0.9, "motion": 0.1}

    cfg = EngineConfig.model_validate(data)
    # Default switching weights are 0.7 / 0.3
    assert cfg.cameras[0].weights.ball == 0.7
    assert cfg.cameras[0].weights.motion == 0.3
    # cam2 keeps its custom weights
    assert cfg.cameras[1].weights.ball == 0.9
    assert cfg.cameras[1].weights.motion == 0.1


def test_log_level_validation():
    valid = LoggingConfig(level="debug")
    assert valid.level == "DEBUG"

    with pytest.raises(ValidationError, match="Invalid log level"):
        LoggingConfig(level="NONEXISTENT")


def test_configuration_error_exception():
    assert issubclass(ConfigurationError, ValueError)
    err = ConfigurationError("Missing scene: cam4")
    assert str(err) == "Missing scene: cam4"
