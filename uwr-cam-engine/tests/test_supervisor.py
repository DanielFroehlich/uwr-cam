"""Unit tests for EngineSupervisor orchestration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import ConfigurationError, EngineConfig, load_config
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
    mock_obs.get_scenes = AsyncMock(return_value=["base", "cam1", "cam2"])
    mock_obs.get_current_scene = AsyncMock(return_value="base")
    mock_obs.set_scene = AsyncMock(return_value=True)
    mock_obs.ensure_text_overlay = AsyncMock(return_value=True)
    mock_obs.update_text_overlay = AsyncMock(return_value=True)
    mock_obs.ensure_marker_source = AsyncMock(return_value=True)
    mock_obs.update_marker_transform = AsyncMock(return_value=True)
    mock_obs.set_marker_enabled = AsyncMock(return_value=True)
    mock_obs.canvas_width = 1920
    mock_obs.canvas_height = 1080
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


@pytest.mark.asyncio
async def test_supervisor_on_config_reload_ball_detector_toggle(mock_supervisor):
    sup, _, _, mock_pipeline_mgr, cfg = mock_supervisor
    await sup.start()

    # Create new config with ball_detector disabled
    new_data = sample_valid_config_dict()
    new_data["ball_detector"] = {"enabled": False}
    new_cfg = EngineConfig.model_validate(new_data)

    await sup.on_config_reload(cfg, new_cfg)

    # update_ball_detector should have been invoked on pipeline_mgr
    mock_pipeline_mgr.update_ball_detector.assert_called_once_with(new_cfg.ball_detector)
    assert sup.config.ball_detector.enabled is False

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_pre_switch_suppressed_on_manual_scene(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, _ = mock_supervisor

    # Start with base scene
    sup.arbiter.current_scene = "base"
    sup.arbiter.previous_scene = "base"
    sup.arbiter.last_switch_time = 0.0

    # Provide CV results that would trigger switch to cam1
    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(score=0.9, motion_score=0.5, ball_confidence=0.9),
        "cam2": MagicMock(score=0.1, motion_score=0.1, ball_confidence=0.1),
    }

    # But OBS currently reports active scene is "interview" (unmanaged manual scene)
    mock_obs.get_current_scene = AsyncMock(return_value="interview")

    sup._running = True
    # Run one cycle of CV loop processing logic
    results = sup.pipeline_mgr.process_tick()
    target_scene = sup.arbiter.evaluate(results)
    assert target_scene == "cam1"  # Arbiter proposed switch

    # Simulate supervisor pre-switch check logic
    active_scene = await sup.obs_client.get_current_scene()
    if active_scene and not sup.arbiter.is_auto_scene(active_scene):
        sup.arbiter.update_current_scene(active_scene)
    else:
        await sup.obs_client.set_scene(target_scene)

    # set_scene must NOT have been called
    mock_obs.set_scene.assert_not_called()
    assert sup.arbiter.current_scene == "interview"
    assert sup.arbiter.is_manual_override is True


@pytest.mark.asyncio
async def test_supervisor_pre_switch_success_on_auto_scene(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, _ = mock_supervisor

    sup.arbiter.current_scene = "base"
    sup.arbiter.previous_scene = "base"
    sup.arbiter.last_switch_time = 0.0

    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(score=0.9, motion_score=0.5, ball_confidence=0.9),
        "cam2": MagicMock(score=0.1, motion_score=0.1, ball_confidence=0.1),
    }

    # OBS confirms active scene is "base" (matches previous_scene)
    mock_obs.get_current_scene = AsyncMock(return_value="base")

    results = sup.pipeline_mgr.process_tick()
    target_scene = sup.arbiter.evaluate(results)
    assert target_scene == "cam1"

    # Pre-switch check logic
    active_scene = await sup.obs_client.get_current_scene()
    if active_scene and not sup.arbiter.is_auto_scene(active_scene):
        sup.arbiter.update_current_scene(active_scene)
    elif active_scene != sup.arbiter.previous_scene:
        sup.arbiter.update_current_scene(active_scene)
    else:
        success = await sup.obs_client.set_scene(target_scene)
        if success:
            sup.arbiter.update_current_scene(target_scene)

    mock_obs.set_scene.assert_awaited_once_with("cam1")
    assert sup.arbiter.current_scene == "cam1"


@pytest.mark.asyncio
async def test_supervisor_obs_scene_sync_loop(mock_supervisor):
    sup, mock_obs, _, _, _ = mock_supervisor

    sup.config.switching.scene_poll_interval_sec = 0.01
    await sup.start()

    assert sup.arbiter.current_scene == "base"
    assert sup.arbiter.is_manual_override is False

    # Simulate human director switching to "replay" in OBS
    mock_obs.get_current_scene = AsyncMock(return_value="replay")
    await asyncio.sleep(0.05)

    assert sup.arbiter.current_scene == "replay"
    assert sup.arbiter.is_manual_override is True

    # Simulate director resuming auto-switching by selecting "cam1" in OBS
    mock_obs.get_current_scene = AsyncMock(return_value="cam1")
    await asyncio.sleep(0.05)

    assert sup.arbiter.current_scene == "cam1"
    assert sup.arbiter.is_manual_override is False

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_overlay_updates(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, _ = mock_supervisor

    sup.config.overlay.enabled = True
    sup.config.overlay.update_interval_sec = 0.01

    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(score=0.88, motion_score=0.30, ball_confidence=0.85),
        "cam2": MagicMock(score=0.20, motion_score=0.15, ball_confidence=0.00),
    }

    await sup.start()
    mock_obs.ensure_text_overlay.assert_awaited_once_with(
        sup.config.overlay.source_name,
        sup.config.switching.fallback_scene,
    )

    await asyncio.sleep(0.05)
    assert mock_obs.update_text_overlay.await_count >= 1

    # Inspect call args
    call_args = mock_obs.update_text_overlay.call_args
    assert call_args[0][0] == sup.config.overlay.source_name
    text_content = call_args[0][1]
    assert "CV SCORES" in text_content
    assert "CAM1: 0.88" in text_content
    assert "CAM2: 0.20" in text_content
    assert call_args[1]["font_settings"]["font"]["face"] == sup.config.overlay.font_face

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_marker_tracking(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, _ = mock_supervisor

    sup.config.marker.enabled = True
    sup.config.marker.update_interval_sec = 0.01
    mock_obs.get_current_scene = AsyncMock(return_value="cam1")

    # Frame with detected ball above confidence threshold
    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(
            score=0.9,
            ball_confidence=0.9,
            ball_bbox=(100, 150, 50, 50),
            frame_width=640,
            frame_height=360,
        ),
    }

    await sup.start()
    await asyncio.sleep(0.05)

    # Marker transform should have been updated
    mock_obs.update_marker_transform.assert_awaited()
    # Marker should have been enabled
    mock_obs.set_marker_enabled.assert_awaited_with("cam1", "Ball_Marker", True)

    # Now simulate frame where ball is lost
    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(
            score=0.2,
            ball_confidence=0.0,
            ball_bbox=None,
            frame_width=640,
            frame_height=360,
        ),
    }
    await asyncio.sleep(0.05)

    # Marker should now be hidden
    mock_obs.set_marker_enabled.assert_awaited_with("cam1", "Ball_Marker", False)

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_marker_not_enabled_below_confidence_threshold(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, _ = mock_supervisor

    sup.config.marker.enabled = True
    sup.config.marker.update_interval_sec = 0.01
    sup.config.ball_detector.confidence_threshold = 0.25
    mock_obs.get_current_scene = AsyncMock(return_value="cam1")

    # Detection confidence (0.15) is BELOW threshold (0.25)
    mock_pipeline_mgr.process_tick.return_value = {
        "cam1": MagicMock(
            score=0.3,
            ball_confidence=0.15,
            ball_bbox=(100, 150, 50, 50),
            frame_width=640,
            frame_height=360,
        ),
    }

    await sup.start()
    await asyncio.sleep(0.05)

    # Marker transform should NOT be updated, and marker must NOT be enabled
    mock_obs.update_marker_transform.assert_not_called()
    mock_obs.set_marker_enabled.assert_not_called()

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_on_config_reload_marker_toggle(mock_supervisor):
    sup, mock_obs, _, _, cfg = mock_supervisor

    sup.config.marker.enabled = True
    await sup.start()

    # Create new config with marker disabled
    new_data = sample_valid_config_dict()
    new_data["marker"] = {"enabled": False}
    new_cfg = EngineConfig.model_validate(new_data)

    await sup.on_config_reload(cfg, new_cfg)

    # set_marker_enabled(False) should have been called across scenes
    assert mock_obs.set_marker_enabled.await_count >= 1
    call_args = mock_obs.set_marker_enabled.call_args[0]
    assert call_args[2] is False

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_start_fails_on_missing_scenes(mock_supervisor):
    sup, mock_obs, mock_watcher, mock_pipeline_mgr, _ = mock_supervisor

    mock_obs.verify_scenes = AsyncMock(return_value=(False, ["cam4"]))
    mock_obs.get_scenes = AsyncMock(return_value=["manual", "cam3", "cam2", "cam1", "base"])

    with pytest.raises(ConfigurationError) as exc_info:
        await sup.start()

    assert "Missing required OBS scenes: ['cam4']" in str(exc_info.value)
    assert "Available scenes in OBS: ['manual', 'cam3', 'cam2', 'cam1', 'base']" in str(exc_info.value)

    # Startup should have aborted cleanly
    assert sup._running is False
    mock_obs.disconnect.assert_awaited_once()
    mock_pipeline_mgr.start_all.assert_not_called()
    mock_watcher.start.assert_not_called()


@pytest.mark.asyncio
async def test_supervisor_on_config_reload_rejected_on_missing_scenes(mock_supervisor):
    sup, mock_obs, _, mock_pipeline_mgr, cfg = mock_supervisor
    await sup.start()

    # Create new config with an additional camera 'cam4'
    new_data = sample_valid_config_dict()
    new_data["cameras"].append(
        {
            "id": "cam4",
            "name": "Surface",
            "rtsp_url": "rtsp://localhost:8554/cam4/low",
            "obs_scene": "cam4",
            "weights": {"ball": 0.5, "motion": 0.5},
        }
    )
    new_cfg = EngineConfig.model_validate(new_data)

    # Configure OBS to reject cam4
    mock_obs.verify_scenes = AsyncMock(return_value=(False, ["cam4"]))
    mock_obs.get_scenes = AsyncMock(return_value=["manual", "cam3", "cam2", "cam1", "base"])

    with pytest.raises(ConfigurationError) as exc_info:
        await sup.on_config_reload(cfg, new_cfg)

    assert "Missing required OBS scenes: ['cam4']" in str(exc_info.value)
    assert "Available scenes in OBS: ['manual', 'cam3', 'cam2', 'cam1', 'base']" in str(exc_info.value)

    # Configuration must NOT be updated; active running config must remain old config
    assert sup.config == cfg
    assert len(sup.config.cameras) == len(cfg.cameras)

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_on_config_reload_rejected_on_obs_connection_failure(mock_supervisor, mocker):
    sup, mock_obs, _, _, cfg = mock_supervisor
    await sup.start()

    # Create new config with different OBS port
    new_data = sample_valid_config_dict()
    new_data["obs"]["port"] = 4459
    new_cfg = EngineConfig.model_validate(new_data)

    # Mock candidate OBS client that fails to connect
    mock_failed_obs = MagicMock()
    mock_failed_obs.connect = AsyncMock(return_value=False)
    mocker.patch("src.main.ObswsPythonClient", return_value=mock_failed_obs)

    with pytest.raises(ConfigurationError) as exc_info:
        await sup.on_config_reload(cfg, new_cfg)

    assert "Failed to connect to OBS Studio" in str(exc_info.value)
    # Old configuration and client must remain active
    assert sup.config == cfg
    assert sup.obs_client == mock_obs

    await sup.stop()


@pytest.mark.asyncio
async def test_supervisor_health_loop_skips_setup_when_scenes_missing(mock_supervisor):
    sup, mock_obs, _, _, _ = mock_supervisor

    sup.config.obs.reconnect_interval_sec = 0.01
    mock_obs.is_connected = False
    mock_obs.connect = AsyncMock(return_value=True)
    mock_obs.verify_scenes = AsyncMock(return_value=(False, ["cam4"]))
    mock_obs.get_scenes = AsyncMock(return_value=["base", "cam1", "cam2"])

    sup._running = True
    health_task = asyncio.create_task(sup._obs_health_loop())

    await asyncio.sleep(0.05)
    sup._running = False
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass

    # Marker and overlay setup should NOT be called on missing scenes
    mock_obs.ensure_text_overlay.assert_not_called()
    mock_obs.ensure_marker_source.assert_not_called()


def test_main_handles_configuration_error(mocker):
    from src.main import main
    mocker.patch(
        "argparse.ArgumentParser.parse_args",
        return_value=MagicMock(config_path="config.yaml", log_level=None),
    )
    mocker.patch(
        "src.main.run_engine",
        side_effect=ConfigurationError("Missing required OBS scenes: ['cam4']"),
    )
    mock_exit = mocker.patch("sys.exit")

    main()

    mock_exit.assert_called_once_with(1)





