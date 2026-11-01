"""Unit tests for Supervisor and dynamic configuration hot-reloading in cam-sim."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.config import (
    CameraConfig,
    CameraFeedsConfig,
    CamSimConfig,
    DefaultsConfig,
    FeedConfig,
    ServerConfig,
    load_config,
)
from src.supervisor import Supervisor


def create_sample_camera(cam_id: str, low_path: str = "/cam1/low", high_path: str = "/cam1/high") -> CameraConfig:
    return CameraConfig(
        id=cam_id,
        name=f"Camera {cam_id}",
        source=f"/videos/{cam_id}.mp4",
        feeds=CameraFeedsConfig(
            low_res=FeedConfig(
                enabled=True,
                path=low_path,
                width=640,
                height=360,
                fps=30,
                bitrate="1000k",
            ),
            high_res=FeedConfig(
                enabled=True,
                path=high_path,
                width=1920,
                height=1080,
                fps=30,
                bitrate="6000k",
            ),
        ),
    )


@pytest.fixture
def base_config() -> CamSimConfig:
    return CamSimConfig(
        version="1.0",
        server=ServerConfig(rtsp_port=8554, bind_address="0.0.0.0"),
        defaults=DefaultsConfig(),
        cameras=[
            create_sample_camera("cam1", "/cam1/low", "/cam1/high"),
            create_sample_camera("cam2", "/cam2/low", "/cam2/high"),
        ],
    )


def test_supervisor_initialization(tmp_path, base_config):
    """Test supervisor initialization with and without config_path."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)
    assert sup.config == base_config
    assert sup.config_path == cfg_file.resolve()
    assert sup._last_mtime is not None

    sup_no_path = Supervisor(config=base_config)
    assert sup_no_path.config_path is None
    assert sup_no_path._last_mtime is None


def test_supervisor_no_config_path_noop(base_config):
    """Check config updates should do nothing if config_path is None."""
    sup = Supervisor(config=base_config)
    sup.check_config_updates()
    assert sup.config == base_config


def test_supervisor_hot_reload_no_file_change(tmp_path, base_config):
    """If mtime has not changed, check_config_updates does nothing."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)
    with patch.object(sup, "apply_config_update") as mock_apply:
        sup.check_config_updates()
        mock_apply.assert_not_called()


def test_supervisor_hot_reload_invalid_yaml(tmp_path, base_config):
    """If file is corrupted with invalid YAML, hot-reload is rejected and active config retained."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    # Corrupt config file
    time.sleep(0.01)
    cfg_file.write_text("invalid: [broken yaml")

    with patch.object(sup, "apply_config_update") as mock_apply:
        sup.check_config_updates()
        mock_apply.assert_not_called()
        assert sup.config == base_config


def test_supervisor_hot_reload_validation_failure(tmp_path, base_config):
    """If config schema is invalid (e.g., fewer than 2 cameras), hot-reload is rejected."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    # Write config with only 1 camera (violates min_length=2)
    invalid_data = base_config.model_dump()
    invalid_data["cameras"] = invalid_data["cameras"][:1]
    time.sleep(0.01)
    cfg_file.write_text(yaml.dump(invalid_data))

    with patch.object(sup, "apply_config_update") as mock_apply:
        sup.check_config_updates()
        mock_apply.assert_not_called()
        assert sup.config == base_config


def test_supervisor_hot_reload_server_changed(tmp_path, base_config):
    """When server config changes, restart MediaMTX and all workers."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    new_data = base_config.model_dump()
    new_data["server"]["rtsp_port"] = 8555
    time.sleep(0.01)
    cfg_file.write_text(yaml.dump(new_data))

    with patch.object(sup, "stop_camera_workers") as mock_stop_workers, \
         patch.object(sup, "stop_mediamtx") as mock_stop_mtx, \
         patch.object(sup, "start_mediamtx") as mock_start_mtx, \
         patch.object(sup, "start_camera_workers") as mock_start_workers:

        sup.check_config_updates()

        mock_stop_workers.assert_called_once()
        mock_stop_mtx.assert_called_once()
        mock_start_mtx.assert_called_once()
        mock_start_workers.assert_called_once()
        assert sup.config.server.rtsp_port == 8555


def test_supervisor_hot_reload_defaults_changed(tmp_path, base_config):
    """When defaults change, restart all camera workers."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    new_data = base_config.model_dump()
    new_data["defaults"]["preset"] = "veryfast"
    time.sleep(0.01)
    cfg_file.write_text(yaml.dump(new_data))

    with patch.object(sup, "stop_camera_workers") as mock_stop_workers, \
         patch.object(sup, "stop_mediamtx") as mock_stop_mtx, \
         patch.object(sup, "start_camera_workers") as mock_start_workers:

        sup.check_config_updates()

        mock_stop_workers.assert_called_once()
        mock_stop_mtx.assert_not_called()
        mock_start_workers.assert_called_once()
        assert sup.config.defaults.preset == "veryfast"


def test_supervisor_hot_reload_surgical_camera_update(tmp_path, base_config):
    """When only one camera changes, only that camera's worker is restarted."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    mock_worker1 = MagicMock()
    mock_worker2 = MagicMock()
    sup.workers = {"cam1": mock_worker1, "cam2": mock_worker2}

    # Modify only cam1 (change bitrate)
    new_data = base_config.model_dump()
    new_data["cameras"][0]["feeds"]["low_res"]["bitrate"] = "2000k"
    time.sleep(0.01)
    cfg_file.write_text(yaml.dump(new_data))

    with patch.object(sup, "_start_camera_worker") as mock_start_single:
        sup.check_config_updates()

        # cam1 worker stopped and restarted
        mock_worker1.stop.assert_called_once()
        # cam2 worker untouched
        mock_worker2.stop.assert_not_called()
        # new cam1 worker started
        mock_start_single.assert_called_once()
        assert mock_start_single.call_args[0][0].id == "cam1"
        assert sup.config.cameras[0].feeds.low_res.bitrate == "2000k"


def test_supervisor_hot_reload_camera_added_and_removed(tmp_path, base_config):
    """Test camera addition and removal during hot-reload."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(base_config.model_dump()))

    sup = Supervisor(config=base_config, config_path=cfg_file)

    mock_worker1 = MagicMock()
    mock_worker2 = MagicMock()
    sup.workers = {"cam1": mock_worker1, "cam2": mock_worker2}

    # Remove cam2, add cam3
    new_data = base_config.model_dump()
    new_data["cameras"] = [
        new_data["cameras"][0],
        create_sample_camera("cam3", "/cam3/low", "/cam3/high").model_dump(),
    ]
    time.sleep(0.01)
    cfg_file.write_text(yaml.dump(new_data))

    with patch.object(sup, "_start_camera_worker") as mock_start_single:
        sup.check_config_updates()

        # cam1 worker untouched
        mock_worker1.stop.assert_not_called()
        # cam2 worker stopped and removed
        mock_worker2.stop.assert_called_once()
        assert "cam2" not in sup.workers
        # cam3 worker started
        mock_start_single.assert_called_once()
        assert mock_start_single.call_args[0][0].id == "cam3"
