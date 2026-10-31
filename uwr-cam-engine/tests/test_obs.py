"""Unit tests for OBS Studio WebSocket client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from obsws_python.error import OBSSDKError, OBSSDKRequestError

from src.config import OBSConfig
from src.obs import OBSClientProtocol, ObswsPythonClient


@pytest.fixture
def obs_config():
    return OBSConfig(
        host="localhost",
        port=4455,
        password="test_password",
        timeout_sec=1.0,
        reconnect_interval_sec=1.0,
    )


def test_protocol_conformance(obs_config):
    client = ObswsPythonClient(obs_config)
    assert isinstance(client, OBSClientProtocol)


@pytest.mark.asyncio
async def test_connect_success(obs_config, mocker):
    mock_req = MagicMock()
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    assert not client.is_connected

    result = await client.connect()
    assert result is True
    assert client.is_connected


@pytest.mark.asyncio
async def test_connect_connection_refused(obs_config, mocker):
    mocker.patch("obsws_python.ReqClient", side_effect=ConnectionRefusedError("Refused"))

    client = ObswsPythonClient(obs_config)
    result = await client.connect()
    assert result is False
    assert not client.is_connected


@pytest.mark.asyncio
async def test_connect_sdk_error(obs_config, mocker):
    mocker.patch("obsws_python.ReqClient", side_effect=OBSSDKError("Auth failed"))

    client = ObswsPythonClient(obs_config)
    result = await client.connect()
    assert result is False
    assert not client.is_connected


@pytest.mark.asyncio
async def test_disconnect(obs_config, mocker):
    mock_req = MagicMock()
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()
    assert client.is_connected

    await client.disconnect()
    assert not client.is_connected
    mock_req.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_get_scenes(obs_config, mocker):
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.scenes = [{"sceneName": "base"}, {"sceneName": "cam1"}, {"sceneName": "cam2"}]
    mock_req.get_scene_list.return_value = mock_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    scenes = await client.get_scenes()
    assert scenes == ["base", "cam1", "cam2"]
    mock_req.get_scene_list.assert_called_once()


@pytest.mark.asyncio
async def test_get_current_scene(obs_config, mocker):
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.current_program_scene_name = "cam1"
    mock_req.get_current_program_scene.return_value = mock_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    current = await client.get_current_scene()
    assert current == "cam1"


@pytest.mark.asyncio
async def test_set_scene_success(obs_config, mocker):
    mock_req = MagicMock()
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    success = await client.set_scene("cam2")
    assert success is True
    mock_req.set_current_program_scene.assert_called_once_with("cam2")


@pytest.mark.asyncio
async def test_set_scene_failure(obs_config, mocker):
    mock_req = MagicMock()
    mock_req.set_current_program_scene.side_effect = OBSSDKRequestError(
        "SetCurrentProgramScene", 600, "No source found by name"
    )
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    success = await client.set_scene("invalid_scene")
    assert success is False


@pytest.mark.asyncio
async def test_verify_scenes_all_present(obs_config, mocker):
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.scenes = [{"sceneName": "base"}, {"sceneName": "cam1"}, {"sceneName": "cam2"}]
    mock_req.get_scene_list.return_value = mock_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    ok, missing = await client.verify_scenes(["base", "cam1"])
    assert ok is True
    assert missing == []


@pytest.mark.asyncio
async def test_verify_scenes_missing(obs_config, mocker):
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.scenes = [{"sceneName": "base"}, {"sceneName": "cam1"}]
    mock_req.get_scene_list.return_value = mock_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    ok, missing = await client.verify_scenes(["base", "cam1", "cam2", "cam3"])
    assert ok is False
    assert missing == ["cam2", "cam3"]


@pytest.mark.asyncio
async def test_ensure_text_overlay_already_exists(obs_config, mocker):
    mock_req = MagicMock()
    mock_inputs_resp = MagicMock()
    mock_inputs_resp.inputs = [{"inputName": "CV_Scores_Overlay"}]
    mock_req.get_input_list.return_value = mock_inputs_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    res = await client.ensure_text_overlay("CV_Scores_Overlay", "base")
    assert res is True
    mock_req.create_input.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_text_overlay_creates_source(obs_config, mocker):
    mock_req = MagicMock()
    mock_inputs_resp = MagicMock()
    mock_inputs_resp.inputs = [{"inputName": "OtherSource"}]
    mock_req.get_input_list.return_value = mock_inputs_resp

    mock_kinds_resp = MagicMock()
    mock_kinds_resp.input_kinds = ["image_source", "text_ft2_source_v2"]
    mock_req.get_input_kind_list.return_value = mock_kinds_resp

    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    res = await client.ensure_text_overlay("CV_Scores_Overlay", "base")
    assert res is True
    mock_req.create_input.assert_called_once_with(
        sceneName="base",
        inputName="CV_Scores_Overlay",
        inputKind="text_ft2_source_v2",
        inputSettings={"text": "Initializing overlay..."},
        sceneItemEnabled=True,
    )


@pytest.mark.asyncio
async def test_update_text_overlay(obs_config, mocker):
    mock_req = MagicMock()
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    font_settings = {"outline": True, "drop_shadow": True}
    res = await client.update_text_overlay("CV_Scores_Overlay", "TEST TEXT", font_settings)
    assert res is True
    mock_req.set_input_settings.assert_called_once_with(
        name="CV_Scores_Overlay",
        settings={"text": "TEST TEXT", "outline": True, "drop_shadow": True},
        overlay=True,
    )


def test_generate_marker_png(tmp_path):
    from src.obs import generate_marker_png
    import cv2
    from pathlib import Path

    target_path = tmp_path / "test_marker.png"
    result = generate_marker_png(str(target_path), color_rgb=(255, 0, 0), thickness=6)
    assert Path(result).is_file()

    img = cv2.imread(result, cv2.IMREAD_UNCHANGED)
    assert img.shape == (256, 256, 4)
    # Check that alpha channel is 255 on border and 0 in center
    assert img[3, 3, 3] == 255
    assert img[128, 128, 3] == 0


@pytest.mark.asyncio
async def test_ensure_marker_source_creates_input(obs_config, mocker):
    mock_req = MagicMock()
    mock_inputs_resp = MagicMock()
    mock_inputs_resp.inputs = []
    mock_req.get_input_list.return_value = mock_inputs_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    res = await client.ensure_marker_source("Ball_Marker", "cam1", "/tmp/marker.png")
    assert res is True
    mock_req.create_input.assert_called_once_with(
        sceneName="cam1",
        inputName="Ball_Marker",
        inputKind="image_source",
        inputSettings={"file": "/tmp/marker.png"},
        sceneItemEnabled=False,
    )


@pytest.mark.asyncio
async def test_update_marker_transform(obs_config, mocker):
    mock_req = MagicMock()
    mock_v = MagicMock()
    mock_v.base_width = 1920
    mock_v.base_height = 1080
    mock_req.get_video_settings.return_value = mock_v
    mock_item_resp = MagicMock()
    mock_item_resp.scene_item_id = 42
    mock_req.get_scene_item_id.return_value = mock_item_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()
    # Client has 1920x1080 canvas
    # Frame is 640x360 (scale factor 3.0)
    # bbox_xywh = (100, 100, 50, 50) -> center=(100, 100) -> top-left=(75, 75)
    # pos_x = 75 * 3.0 = 225.0, pos_y = 75 * 3.0 = 225.0
    # target_w = 50 * 3.0 = 150.0, target_h = 50 * 3.0 = 150.0
    # scale = 150.0 / 256.0
    res = await client.update_marker_transform(
        scene_name="cam1",
        source_name="Ball_Marker",
        bbox_xywh=(100, 100, 50, 50),
        frame_w=640,
        frame_h=360,
    )
    assert res is True
    mock_req.set_scene_item_transform.assert_called_once()
    call_args = mock_req.set_scene_item_transform.call_args[1]
    assert call_args["scene_name"] == "cam1"
    assert call_args["item_id"] == 42
    transform = call_args["transform"]
    assert transform["positionX"] == 225.0
    assert transform["positionY"] == 225.0
    assert transform["scaleX"] == 150.0 / 256.0


@pytest.mark.asyncio
async def test_set_marker_enabled(obs_config, mocker):
    mock_req = MagicMock()
    mock_item_resp = MagicMock()
    mock_item_resp.scene_item_id = 42
    mock_req.get_scene_item_id.return_value = mock_item_resp
    mocker.patch("obsws_python.ReqClient", return_value=mock_req)

    client = ObswsPythonClient(obs_config)
    await client.connect()

    res = await client.set_marker_enabled("cam1", "Ball_Marker", True)
    assert res is True
    mock_req.set_scene_item_enabled.assert_called_once_with(
        scene_name="cam1", item_id=42, enabled=True
    )


