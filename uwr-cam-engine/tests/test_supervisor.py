"""Unit tests for EngineSupervisor orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import EngineConfig, load_config
from src.main import EngineSupervisor
from tests.test_config import sample_valid_config_dict


@pytest.fixture
def mock_supervisor(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    data = sample_valid_config_dict()
    import yaml
    config_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(config_file)

    # Mock ObswsPythonClient
    mock_obs = MagicMock()
    mock_obs.connect = AsyncMock(return_value=True)
    mock_obs.disconnect = AsyncMock()
    mock_obs.verify_scenes = AsyncMock(return_value=(True, []))
    mock_obs.get_current_scene = AsyncMock(return_value="base")
    mock_obs.set_scene = AsyncMock(return_value=True)
    mock_obs.is_connected = True
    mocker.patch("src.main.ObswsPythonClient", return_value=mock_obs)

    # Mock ConfigWatcher
    mock_watcher = MagicMock()
    mock_watcher.start = AsyncMock()
    mock_watcher.stop = AsyncMock()
    mock_watcher.add_callback = MagicMock()
    mocker.patch("src.main.ConfigWatcher", return_value=mock_watcher)

    # Mock PipelineManager
    mock_pipeline_mgr = MagicMock()
    mock_pipeline_mgr.start_all = MagicMock()
    mock_pipeline_mgr.stop_all = MagicMock()
    mock_pipeline_mgr.process_tick = MagicMock(return_value={})
    mocker.patch("src.main.PipelineManager", return_value=mock_pipeline_mgr)

    sup = EngineSupervisor(config_file, cfg)
    return sup, mock_obs, mock_watcher, mock_pipeline_mgr, cfg


@pytest.mark.asyncio
async def test_supervisor_start_and_stop(mock_supervisor):
    sup, mock_obs, mock_watcher, mock_pipeline_mgr, _ = mock_supervisor

    await sup.start()
    assert sup._running is True
    mock_obs.connect.assert_awaited_once()
    mock_obs.verify_scenes.assert_awaited_once()
    mock_watcher.start.assert_awaited_once()
    mock_pipeline_mgr.start_all.assert_called_once()

    await sup.stop()
    assert sup._running is False
    mock_watcher.stop.assert_awaited_once()
    mock_obs.disconnect.assert_awaited_once()
    mock_pipeline_mgr.stop_all.assert_called_once()


@pytest.mark.asyncio
async def test_supervisor_on_config_reload_obs_change(mock_supervisor, mocker):
    sup, mock_obs, _, _, cfg = mock_supervisor
    await sup.start()

    # Create new config with different OBS port
    new_data = sample_valid_config_dict()
    new_data["obs"]["port"] = 4456
    new_cfg = EngineConfig.model_validate(new_data)

    mock_new_obs = MagicMock()
    mock_new_obs.connect = AsyncMock(return_value=True)
    mock_new_obs.disconnect = AsyncMock()
    mock_new_obs.verify_scenes = AsyncMock(return_value=(True, []))
    mock_new_obs.get_current_scene = AsyncMock(return_value="base")
    mock_new_obs.set_scene = AsyncMock(return_value=True)
    mock_new_obs.is_connected = True
    mocker.patch("src.main.ObswsPythonClient", return_value=mock_new_obs)

    await sup.on_config_reload(cfg, new_cfg)

    # Old client should have disconnected
    mock_obs.disconnect.assert_awaited_once()
    # New client should have connected and verified scenes
    mock_new_obs.connect.assert_awaited_once()
    mock_new_obs.verify_scenes.assert_awaited_once()

    await sup.stop()
