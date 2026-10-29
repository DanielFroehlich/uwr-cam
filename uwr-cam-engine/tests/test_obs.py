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
