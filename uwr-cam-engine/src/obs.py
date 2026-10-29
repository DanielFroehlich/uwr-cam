"""OBS Studio WebSocket v5 client interface and obsws-python implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import obsws_python as obs
from obsws_python.error import OBSSDKError, OBSSDKRequestError

from .config import OBSConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class OBSClientProtocol(Protocol):
    """Protocol defining the OBS Studio WebSocket controller interface.

    Allows seamless swapping between obsws-python and alternative async
    websocket implementations.
    """

    @property
    def is_connected(self) -> bool:
        """Return True if currently connected to OBS WebSocket."""
        ...

    async def connect(self) -> bool:
        """Connect to OBS Studio. Returns True if successful, False otherwise."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from OBS Studio cleanly."""
        ...

    async def get_scenes(self) -> List[str]:
        """Fetch list of all available scene names in OBS."""
        ...

    async def get_current_scene(self) -> Optional[str]:
        """Fetch name of the currently active program scene."""
        ...

    async def set_scene(self, scene_name: str) -> bool:
        """Switch current program scene to scene_name. Returns True if successful."""
        ...

    async def verify_scenes(self, required_scenes: List[str]) -> Tuple[bool, List[str]]:
        """Verify that all required scenes exist in OBS.

        Returns:
            Tuple of (all_exist_bool, list_of_missing_scenes)
        """
        ...

    async def ensure_connected(self) -> bool:
        """Ensure connection is active, attempting reconnection if disconnected."""
        ...


class ObswsPythonClient:
    """OBS WebSocket v5 client implementing OBSClientProtocol using obsws-python.

    All underlying blocking network calls are delegated to background threads via
    asyncio.to_thread, guaranteeing that the central asyncio loop remains responsive
    and non-blocking.
    """

    def __init__(self, config: OBSConfig) -> None:
        self.config = config
        self._client: Optional[obs.ReqClient] = None
        self._connected: bool = False
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _sync_connect(self) -> obs.ReqClient:
        """Blocking connect function to run in worker thread."""
        logger.debug(f"Attempting connection to OBS at {self.config.host}:{self.config.port}...")
        client = obs.ReqClient(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            timeout=int(self.config.timeout_sec),
        )
        return client

    async def connect(self) -> bool:
        """Establish connection to OBS Studio WebSocket server."""
        async with self._lock:
            if self._connected and self._client is not None:
                return True

            try:
                self._client = await asyncio.to_thread(self._sync_connect)
                self._connected = True
                logger.info(
                    f"Successfully connected to OBS Studio WebSocket at {self.config.host}:{self.config.port}"
                )
                return True
            except ConnectionRefusedError:
                logger.warning(
                    f"Connection refused by OBS WebSocket at {self.config.host}:{self.config.port}. "
                    "Ensure OBS Studio is running with WebSocket enabled."
                )
                self._client = None
                self._connected = False
                return False
            except OBSSDKError as e:
                logger.error(f"OBS WebSocket SDK error during connection: {e}")
                self._client = None
                self._connected = False
                return False
            except Exception as e:
                logger.error(f"Failed to connect to OBS WebSocket: {e}")
                self._client = None
                self._connected = False
                return False

    def _sync_disconnect(self, client: obs.ReqClient) -> None:
        try:
            client.disconnect()
        except Exception as e:
            logger.debug(f"Error during OBS client disconnect: {e}")

    async def disconnect(self) -> None:
        """Disconnect cleanly from OBS Studio."""
        async with self._lock:
            client = self._client
            self._client = None
            self._connected = False
            if client is not None:
                await asyncio.to_thread(self._sync_disconnect, client)
                logger.info("Disconnected from OBS Studio WebSocket")

    def _sync_get_scenes(self, client: obs.ReqClient) -> List[str]:
        resp = client.get_scene_list()
        # resp.scenes is a list of dicts: [{'sceneIndex': 0, 'sceneName': 'base'}, ...]
        scenes: List[str] = []
        for s in resp.scenes:
            if isinstance(s, dict) and "sceneName" in s:
                scenes.append(s["sceneName"])
            elif hasattr(s, "scene_name"):
                scenes.append(getattr(s, "scene_name"))
            elif hasattr(s, "sceneName"):
                scenes.append(getattr(s, "sceneName"))
        return scenes

    async def get_scenes(self) -> List[str]:
        """Fetch all available scenes in OBS."""
        if not await self.ensure_connected():
            return []

        assert self._client is not None
        try:
            scenes = await asyncio.to_thread(self._sync_get_scenes, self._client)
            return scenes
        except Exception as e:
            logger.error(f"Failed to get OBS scene list: {e}")
            await self._handle_disconnect()
            return []

    def _sync_get_current_scene(self, client: obs.ReqClient) -> str:
        resp = client.get_current_program_scene()
        return resp.current_program_scene_name

    async def get_current_scene(self) -> Optional[str]:
        """Fetch current active program scene name."""
        if not await self.ensure_connected():
            return None

        assert self._client is not None
        try:
            scene_name = await asyncio.to_thread(self._sync_get_current_scene, self._client)
            return scene_name
        except Exception as e:
            logger.error(f"Failed to get current OBS program scene: {e}")
            await self._handle_disconnect()
            return None

    def _sync_set_scene(self, client: obs.ReqClient, scene_name: str) -> None:
        client.set_current_program_scene(scene_name)

    async def set_scene(self, scene_name: str) -> bool:
        """Switch program scene to target scene."""
        if not await self.ensure_connected():
            logger.warning(f"Cannot switch to scene '{scene_name}': not connected to OBS")
            return False

        assert self._client is not None
        try:
            await asyncio.to_thread(self._sync_set_scene, self._client, scene_name)
            logger.info(f"OBS Program scene switched to: '{scene_name}'")
            return True
        except OBSSDKRequestError as e:
            logger.error(f"OBS rejected scene switch to '{scene_name}': {e}")
            return False
        except Exception as e:
            logger.error(f"Exception switching OBS scene to '{scene_name}': {e}")
            await self._handle_disconnect()
            return False

    async def verify_scenes(self, required_scenes: List[str]) -> Tuple[bool, List[str]]:
        """Verify that all required scenes exist in OBS Studio."""
        available = await self.get_scenes()
        if not available and not self.is_connected:
            return False, required_scenes

        missing = [scene for scene in required_scenes if scene not in available]
        if missing:
            logger.warning(
                f"Missing required OBS scenes: {missing}. Available scenes in OBS: {available}"
            )
            return False, missing

        logger.info(f"Verified all required scenes exist in OBS: {required_scenes}")
        return True, []

    async def _handle_disconnect(self) -> None:
        """Mark as disconnected on network or RPC errors."""
        async with self._lock:
            self._connected = False
            client = self._client
            self._client = None
            if client is not None:
                await asyncio.to_thread(self._sync_disconnect, client)

    async def ensure_connected(self) -> bool:
        """Check connection state, attempting to reconnect if disconnected."""
        if self.is_connected:
            return True
        return await self.connect()
