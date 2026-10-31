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

    async def ensure_text_overlay(self, source_name: str, scene_name: str) -> bool:
        """Ensure the specified text overlay input exists in OBS, creating it if needed."""
        ...

    async def update_text_overlay(
        self, source_name: str, text: str, font_settings: Optional[dict] = None
    ) -> bool:
        """Update the text and formatting of an existing text overlay in OBS."""
        ...


    @property
    def canvas_width(self) -> int:
        """Base canvas width of OBS Studio."""
        ...

    @property
    def canvas_height(self) -> int:
        """Base canvas height of OBS Studio."""
        ...

    async def ensure_marker_source(
        self, source_name: str, scene_name: str, image_path: str
    ) -> bool:
        """Ensure the specified marker image source exists in OBS, creating it if needed."""
        ...

    async def update_marker_transform(
        self,
        scene_name: str,
        source_name: str,
        bbox_xywh: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Update position and scale of the marker based on normalized bounding box."""
        ...

    async def set_marker_enabled(
        self, scene_name: str, source_name: str, enabled: bool
    ) -> bool:
        """Toggle visibility of the marker scene item in the given scene."""
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
        self._canvas_width: int = 1920
        self._canvas_height: int = 1080

    @property
    def canvas_width(self) -> int:
        return self._canvas_width

    @property
    def canvas_height(self) -> int:
        return self._canvas_height

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _sync_connect(self) -> Tuple[obs.ReqClient, int, int]:
        """Blocking connect function to run in worker thread."""
        logger.debug(f"Attempting connection to OBS at {self.config.host}:{self.config.port}...")
        client = obs.ReqClient(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            timeout=int(self.config.timeout_sec),
        )
        cw, ch = 1920, 1080
        try:
            v_settings = client.get_video_settings()
            cw = getattr(v_settings, "base_width", 1920)
            ch = getattr(v_settings, "base_height", 1080)
        except Exception as e:
            logger.debug(f"Could not query OBS video settings, defaulting to 1920x1080: {e}")
        return client, cw, ch

    async def connect(self) -> bool:
        """Establish connection to OBS Studio WebSocket server."""
        async with self._lock:
            if self._connected and self._client is not None:
                return True

            try:
                self._client, self._canvas_width, self._canvas_height = await asyncio.to_thread(self._sync_connect)
                self._connected = True
                logger.info(
                    f"Successfully connected to OBS Studio WebSocket at {self.config.host}:{self.config.port} "
                    f"(Canvas: {self._canvas_width}x{self._canvas_height})"
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
            logger.debug(f"OBS Program scene switched to: '{scene_name}'")
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
            logger.error(
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

    def _sync_ensure_text_overlay(
        self, client: obs.ReqClient, source_name: str, scene_name: str
    ) -> bool:
        inputs = client.get_input_list().inputs
        input_names = [
            i["inputName"] if isinstance(i, dict) else getattr(i, "inputName", getattr(i, "input_name", ""))
            for i in inputs
        ]
        if source_name in input_names:
            logger.debug(f"OBS text overlay input '{source_name}' already exists.")
            return True

        # Find appropriate text input kind
        kinds = client.get_input_kind_list(unversioned=False).input_kinds
        if "text_ft2_source_v2" in kinds:
            kind = "text_ft2_source_v2"
        elif "text_gdiplus_v2" in kinds:
            kind = "text_gdiplus_v2"
        elif "text_ft2_source" in kinds:
            kind = "text_ft2_source"
        elif "text_gdiplus" in kinds:
            kind = "text_gdiplus"
        else:
            logger.error("No compatible text input kind found in OBS for overlay.")
            return False

        client.create_input(
            sceneName=scene_name,
            inputName=source_name,
            inputKind=kind,
            inputSettings={"text": "Initializing overlay..."},
            sceneItemEnabled=True,
        )
        logger.info(f"Created OBS text overlay input '{source_name}' ({kind}) in scene '{scene_name}'")
        return True

    async def ensure_text_overlay(self, source_name: str, scene_name: str) -> bool:
        """Ensure the specified text overlay input exists in OBS, creating it if needed."""
        if not await self.ensure_connected():
            return False

        assert self._client is not None
        try:
            return await asyncio.to_thread(
                self._sync_ensure_text_overlay, self._client, source_name, scene_name
            )
        except Exception as e:
            logger.error(f"Failed to ensure text overlay source '{source_name}': {e}")
            return False

    def _sync_update_text_overlay(
        self,
        client: obs.ReqClient,
        source_name: str,
        text: str,
        font_settings: Optional[dict] = None,
    ) -> bool:
        settings: dict = {"text": text}
        if font_settings:
            settings.update(font_settings)

        client.set_input_settings(name=source_name, settings=settings, overlay=True)
        return True

    async def update_text_overlay(
        self, source_name: str, text: str, font_settings: Optional[dict] = None
    ) -> bool:
        """Update the text and formatting of an existing text overlay in OBS."""
        if not await self.ensure_connected():
            return False

        assert self._client is not None
        try:
            return await asyncio.to_thread(
                self._sync_update_text_overlay,
                self._client,
                source_name,
                text,
                font_settings,
            )
        except Exception as e:
            logger.debug(f"Failed to update text overlay '{source_name}': {e}")
            return False

    def _sync_ensure_marker_source(
        self, client: obs.ReqClient, source_name: str, scene_name: str, image_path: str
    ) -> bool:
        inputs = client.get_input_list().inputs
        input_names = [
            i["inputName"] if isinstance(i, dict) else getattr(i, "inputName", getattr(i, "input_name", ""))
            for i in inputs
        ]

        if source_name not in input_names:
            client.create_input(
                sceneName=scene_name,
                inputName=source_name,
                inputKind="image_source",
                inputSettings={"file": image_path},
                sceneItemEnabled=False,
            )
            logger.info(f"Created OBS marker input '{source_name}' in scene '{scene_name}'")
            return True

        # Input exists globally. Ensure it is present as a scene item in target scene
        try:
            client.get_scene_item_id(scene_name=scene_name, source_name=source_name)
        except Exception:
            # Not in this scene, add it
            try:
                client.create_scene_item(scene_name=scene_name, source_name=source_name, enabled=False)
                logger.debug(f"Added marker scene item '{source_name}' to scene '{scene_name}'")
            except Exception as e:
                logger.debug(f"Could not add marker scene item to '{scene_name}': {e}")
        return True

    async def ensure_marker_source(
        self, source_name: str, scene_name: str, image_path: str
    ) -> bool:
        """Ensure the specified marker image source exists in OBS, creating it if needed."""
        if not await self.ensure_connected():
            return False

        assert self._client is not None
        try:
            return await asyncio.to_thread(
                self._sync_ensure_marker_source,
                self._client,
                source_name,
                scene_name,
                image_path,
            )
        except Exception as e:
            logger.error(f"Failed to ensure marker source '{source_name}': {e}")
            return False

    def _sync_update_marker_transform(
        self,
        client: obs.ReqClient,
        scene_name: str,
        source_name: str,
        bbox_xywh: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        item_resp = client.get_scene_item_id(scene_name=scene_name, source_name=source_name)
        item_id = item_resp.scene_item_id

        xc, yc, w, h = bbox_xywh
        scale_x = self._canvas_width / max(1, frame_w)
        scale_y = self._canvas_height / max(1, frame_h)

        pos_x = (xc - (w / 2.0)) * scale_x
        pos_y = (yc - (h / 2.0)) * scale_y
        target_w = w * scale_x
        target_h = h * scale_y

        # Marker asset original size is 256x256
        scale_item_x = target_w / 256.0
        scale_item_y = target_h / 256.0

        client.set_scene_item_transform(
            scene_name=scene_name,
            item_id=item_id,
            transform={
                "positionX": pos_x,
                "positionY": pos_y,
                "scaleX": scale_item_x,
                "scaleY": scale_item_y,
                "boundsType": "OBS_BOUNDS_NONE",
            },
        )
        return True

    async def update_marker_transform(
        self,
        scene_name: str,
        source_name: str,
        bbox_xywh: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """Update position and scale of the marker based on normalized bounding box."""
        if not await self.ensure_connected():
            return False

        assert self._client is not None
        try:
            return await asyncio.to_thread(
                self._sync_update_marker_transform,
                self._client,
                scene_name,
                source_name,
                bbox_xywh,
                frame_w,
                frame_h,
            )
        except Exception as e:
            logger.debug(f"Failed to update marker transform: {e}")
            return False

    def _sync_set_marker_enabled(
        self, client: obs.ReqClient, scene_name: str, source_name: str, enabled: bool
    ) -> bool:
        item_resp = client.get_scene_item_id(scene_name=scene_name, source_name=source_name)
        item_id = item_resp.scene_item_id
        client.set_scene_item_enabled(scene_name=scene_name, item_id=item_id, enabled=enabled)
        return True

    async def set_marker_enabled(
        self, scene_name: str, source_name: str, enabled: bool
    ) -> bool:
        """Toggle visibility of the marker scene item in the given scene."""
        if not await self.ensure_connected():
            return False

        assert self._client is not None
        try:
            return await asyncio.to_thread(
                self._sync_set_marker_enabled,
                self._client,
                scene_name,
                source_name,
                enabled,
            )
        except Exception as e:
            logger.debug(f"Failed to set marker enabled: {e}")
            return False


def generate_marker_png(
    file_path: str, color_rgb: Tuple[int, int, int] = (0, 255, 0), thickness: int = 4
) -> str:
    """Generate a transparent PNG with a hollow rectangular border."""
    import cv2
    import numpy as np
    from pathlib import Path

    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    size = 256
    img = np.zeros((size, size, 4), dtype=np.uint8)
    b, g, r = color_rgb[2], color_rgb[1], color_rgb[0]
    half_th = thickness // 2
    cv2.rectangle(
        img,
        (half_th, half_th),
        (size - 1 - half_th, size - 1 - half_th),
        (b, g, r, 255),
        thickness=thickness,
    )
    cv2.imwrite(str(p), img)
    logger.info(f"Generated ball marker asset at: {p.resolve()}")
    return str(p.resolve())


