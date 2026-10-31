"""CLI entrypoint and central orchestration loop for uwr-cam-engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .arbiter import SwitchingArbiter
from .config import ConfigurationError, EngineConfig, load_config
from .config_watcher import ConfigWatcher
from .obs import OBSClientProtocol, ObswsPythonClient, generate_marker_png
from .pipeline_manager import CVResult, PipelineManager

logger = logging.getLogger("uwr-cam-engine")


def setup_logging(level_name: str = "INFO") -> None:
    """Configure structured console logging."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def find_default_config() -> str:
    """Locate the default config file across standard locations."""
    candidates = [
        Path("/app/config.yaml"),
        Path("uwr-cam-engine/config.yaml"),
        Path("config.yaml"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return "config.yaml"


class EngineSupervisor:
    """Manages the lifecycle of uwr-cam-engine services."""

    def __init__(self, config_path: str | Path, config: EngineConfig) -> None:
        self.config_path = Path(config_path)
        self.config = config
        self.obs_client: OBSClientProtocol = ObswsPythonClient(self.config.obs)
        self.watcher: ConfigWatcher = ConfigWatcher(self.config_path, self.config)
        self.pipeline_mgr: PipelineManager = PipelineManager(
            self.config.cameras, ball_detector_config=self.config.ball_detector
        )
        self.arbiter: SwitchingArbiter = SwitchingArbiter(
            self.config.switching, self.config.cameras
        )
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._sync_task: Optional[asyncio.Task] = None
        self._cv_task: Optional[asyncio.Task] = None
        self._last_status_log_time: float = 0.0
        self._last_overlay_update_time: float = 0.0
        self._last_marker_update_time: float = 0.0
        self._marker_image_path: Optional[str] = None
        self._marker_visible: bool = False

    async def start(self) -> None:
        """Start all engine background services."""
        self._running = True
        logger.info("Starting uwr-cam-engine...")

        # 1. Connect to OBS Studio
        connected = await self.obs_client.connect()
        if connected:
            required_scenes = [c.obs_scene for c in self.config.cameras] + [
                self.config.switching.fallback_scene
            ]
            ok, missing = await self.obs_client.verify_scenes(required_scenes)
            if not ok:
                available = await self.obs_client.get_scenes()
                err_msg = (
                    f"Missing required OBS scenes: {missing}. Available scenes in OBS: {available}. "
                    "Please ensure all camera scenes and the fallback scene exist in OBS Studio."
                )
                logger.error(f"Startup aborted: {err_msg}")
                await self.obs_client.disconnect()
                self._running = False
                raise ConfigurationError(err_msg)

            active_scene = await self.obs_client.get_current_scene()
            if active_scene:
                self.arbiter.update_current_scene(active_scene)

            if self.config.overlay.enabled:
                await self.obs_client.ensure_text_overlay(
                    self.config.overlay.source_name,
                    self.config.switching.fallback_scene,
                )

            if self.config.marker.enabled:
                await self._init_marker_sources()
        else:
            logger.warning(
                "Initial connection to OBS failed. Background health monitor will attempt reconnection."
            )

        # 2. Start Video Ingestion & CV Pipelines
        self.pipeline_mgr.start_all()

        # 3. Attach hot-reload callback & start config watcher
        self.watcher.add_callback(self.on_config_reload)
        await self.watcher.start()

        # 4. Start background loops
        self._health_task = asyncio.create_task(
            self._obs_health_loop(), name="obs-health-monitor"
        )
        self._sync_task = asyncio.create_task(
            self._obs_scene_sync_loop(), name="obs-scene-sync"
        )
        self._cv_task = asyncio.create_task(
            self._cv_processing_loop(), name="cv-processing-loop"
        )
        logger.info("uwr-cam-engine is running and ready.")

    async def stop(self) -> None:
        """Gracefully stop all services."""
        if not self._running:
            return
        logger.info("Shutting down uwr-cam-engine...")
        self._running = False

        if self._cv_task and not self._cv_task.done():
            self._cv_task.cancel()
            try:
                await self._cv_task
            except asyncio.CancelledError:
                pass

        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        self.pipeline_mgr.stop_all()
        await self.watcher.stop()
        await self.obs_client.disconnect()
        logger.info("uwr-cam-engine stopped cleanly.")

    async def on_config_reload(self, old_cfg: EngineConfig, new_cfg: EngineConfig) -> None:
        """Handle dynamic configuration hot-reloading."""
        logger.info("Validating dynamic configuration updates...")

        # 1. Validate OBS connection and scenes BEFORE applying any changes
        client_to_check: OBSClientProtocol = self.obs_client
        temp_client: Optional[OBSClientProtocol] = None

        if old_cfg.obs != new_cfg.obs:
            logger.info("OBS connection parameters changed. Testing new OBS connection...")
            temp_client = ObswsPythonClient(new_cfg.obs)
            connected = await temp_client.connect()
            if not connected:
                err_msg = (
                    f"Failed to connect to OBS Studio with updated connection parameters: "
                    f"host={new_cfg.obs.host} port={new_cfg.obs.port}."
                )
                logger.error(f"Hot-reload rejected: {err_msg} Retaining active configuration.")
                raise ConfigurationError(err_msg)
            client_to_check = temp_client

        if client_to_check.is_connected:
            required_scenes = [c.obs_scene for c in new_cfg.cameras] + [
                new_cfg.switching.fallback_scene
            ]
            ok, missing = await client_to_check.verify_scenes(required_scenes)
            if not ok:
                available = await client_to_check.get_scenes()
                if temp_client:
                    await temp_client.disconnect()
                err_msg = (
                    f"Missing required OBS scenes: {missing}. Available scenes in OBS: {available}. "
                    "Please ensure all camera scenes and the fallback scene exist in OBS Studio."
                )
                logger.error(f"Hot-reload rejected: {err_msg} Retaining active configuration.")
                raise ConfigurationError(err_msg)

        if temp_client:
            await self.obs_client.disconnect()
            self.obs_client = temp_client

        # 2. All validations passed! Apply runtime updates.
        logger.info("Applying dynamic configuration updates...")
        self.config = new_cfg

        # Check log level change
        if old_cfg.logging.level != new_cfg.logging.level:
            setup_logging(new_cfg.logging.level)
            logger.info(f"Updated log level to {new_cfg.logging.level}")

        # Check if ball detector configuration changed (updates dynamically without stream restart)
        if old_cfg.ball_detector != new_cfg.ball_detector:
            self.pipeline_mgr.update_ball_detector(new_cfg.ball_detector)

        # Check if camera stream endpoints or camera list changed
        old_urls = {c.id: c.rtsp_url for c in old_cfg.cameras}
        new_urls = {c.id: c.rtsp_url for c in new_cfg.cameras}
        if old_urls != new_urls or set(old_urls.keys()) != set(new_urls.keys()):
            logger.info("Camera stream endpoints changed. Reconnecting video ingestion pipelines...")
            self.pipeline_mgr.stop_all()
            self.pipeline_mgr = PipelineManager(
                new_cfg.cameras, ball_detector_config=new_cfg.ball_detector
            )
            self.pipeline_mgr.start_all()
        elif old_cfg.cameras != new_cfg.cameras:
            # Only weights or metadata changed - update weights without dropping streams
            self.pipeline_mgr.update_camera_weights(new_cfg.cameras)

        # Update arbiter with new switching and camera configs
        self.arbiter = SwitchingArbiter(
            new_cfg.switching, new_cfg.cameras, initial_scene=self.arbiter.current_scene
        )

        # Check marker enabled/disabled changes
        if old_cfg.marker.enabled and not new_cfg.marker.enabled:
            logger.info("Ball marker disabled dynamically. Hiding marker across scenes...")
            all_scenes = [c.obs_scene for c in new_cfg.cameras] + [new_cfg.switching.fallback_scene]
            for sc in all_scenes:
                await self.obs_client.set_marker_enabled(sc, old_cfg.marker.source_name, False)
            self._marker_visible = False
        elif not old_cfg.marker.enabled and new_cfg.marker.enabled:
            logger.info("Ball marker enabled dynamically. Initializing marker...")
            await self._init_marker_sources()

        logger.info(
            f"Configuration reloaded: {len(new_cfg.cameras)} cameras active, "
            f"fallback scene: '{new_cfg.switching.fallback_scene}'."
        )

    async def _init_marker_sources(self) -> None:
        """Create marker asset and ensure marker input exists across camera scenes."""
        if not self._marker_image_path:
            self._marker_image_path = generate_marker_png(
                "/tmp/uwr_ball_marker.png",
                color_rgb=tuple(self.config.marker.border_color),
                thickness=self.config.marker.border_thickness,
            )
        all_scenes = [c.obs_scene for c in self.config.cameras] + [
            self.config.switching.fallback_scene
        ]
        for sc in all_scenes:
            await self.obs_client.ensure_marker_source(
                self.config.marker.source_name, sc, self._marker_image_path
            )

    async def _cv_processing_loop(self) -> None:
        """Main async loop ticking at ~30 FPS to process CV, arbitrate, and report status."""
        loop = asyncio.get_running_loop()
        interval = 1.0 / 30.0  # Target ~30 FPS
        self._last_status_log_time = loop.time()

        while self._running:
            t0 = loop.time()
            try:
                # 1. Run the CV frame processing in threadpool to avoid blocking event loop
                results = await asyncio.to_thread(self.pipeline_mgr.process_tick)

                # 2. Evaluate switching arbitration
                target_scene = self.arbiter.evaluate(results)
                if target_scene:
                    # Pre-switch verification: check active OBS scene immediately before switching
                    active_scene = await self.obs_client.get_current_scene()
                    if not active_scene:
                        logger.warning(
                            f"Unable to verify current OBS scene before switch. Aborting switch to '{target_scene}'."
                        )
                        self.arbiter.update_current_scene(self.arbiter.previous_scene)
                    elif not self.arbiter.is_auto_scene(active_scene):
                        logger.info(
                            f"Director manual override detected: OBS program scene is manual scene '{active_scene}'. "
                            f"Suppressing switch to '{target_scene}'."
                        )
                        self.arbiter.update_current_scene(active_scene)
                    elif active_scene != self.arbiter.previous_scene:
                        logger.info(
                            f"OBS active scene changed to '{active_scene}' before switch. "
                            f"Aborting switch to '{target_scene}'."
                        )
                        self.arbiter.update_current_scene(active_scene)
                    else:
                        logger.debug(f"Arbiter commanded scene switch to: '{target_scene}'")
                        success = await self.obs_client.set_scene(target_scene)
                        if success:
                            self.arbiter.update_current_scene(target_scene)
                        else:
                            logger.warning(f"Failed to execute scene switch to '{target_scene}'")
                            self.arbiter.update_current_scene(self.arbiter.previous_scene)

                # Granular debug logging
                logger.debug(f"CV Tick results: {results}")

                # Periodic summary logging at INFO level
                now = loop.time()
                status_interval = self.config.logging.status_interval_sec
                if now - self._last_status_log_time >= status_interval:
                    self._last_status_log_time = now
                    summary_parts = [
                        f"{cam_id}: [score={res.score:.2f}, motion={res.motion_score:.2f}, ball={res.ball_confidence:.2f}]"
                        for cam_id, res in results.items()
                    ]
                    active_scene = self.arbiter.current_scene
                    override_str = " [MANUAL]" if self.arbiter.is_manual_override else ""
                    logger.info(
                        f"[Status] Active: '{active_scene}'{override_str} | Cameras: "
                        + ", ".join(summary_parts)
                    )

                # Periodic overlay update
                if self.config.overlay.enabled and self.obs_client.is_connected:
                    overlay_interval = self.config.overlay.update_interval_sec
                    if now - self._last_overlay_update_time >= overlay_interval:
                        self._last_overlay_update_time = now
                        overlay_lines = [
                            f"CV SCORES [Active: {self.arbiter.current_scene}]"
                        ]
                        for cam in self.config.cameras:
                            cid = cam.id
                            if cid in results:
                                res = results[cid]
                                overlay_lines.append(
                                    f"{cid.upper()}: {res.score:.2f} [Ball: {res.ball_confidence:.2f}, Motion: {res.motion_score:.2f}]"
                                )
                            else:
                                overlay_lines.append(f"{cid.upper()}: --")

                        overlay_text = "\n".join(overlay_lines)
                        font_settings = {
                            "font": {
                                "face": self.config.overlay.font_face,
                                "flags": 0,
                                "size": self.config.overlay.font_size,
                                "style": self.config.overlay.font_style,
                            },
                            "outline": self.config.overlay.outline,
                            "drop_shadow": self.config.overlay.drop_shadow,
                        }
                        await self.obs_client.update_text_overlay(
                            self.config.overlay.source_name,
                            overlay_text,
                            font_settings=font_settings,
                        )

                # Periodic ball marker update
                if self.config.marker.enabled and self.obs_client.is_connected:
                    marker_interval = self.config.marker.update_interval_sec
                    if now - self._last_marker_update_time >= marker_interval:
                        self._last_marker_update_time = now
                        active_scene = self.arbiter.current_scene
                        cam_id = self.arbiter.scene_to_cam.get(active_scene)
                        if cam_id and cam_id in results:
                            res = results[cam_id]
                            min_conf = self.config.ball_detector.confidence_threshold
                            if res.ball_bbox is not None and res.ball_confidence >= min_conf:
                                await self.obs_client.update_marker_transform(
                                    active_scene,
                                    self.config.marker.source_name,
                                    res.ball_bbox,
                                    res.frame_width,
                                    res.frame_height,
                                )
                                if not self._marker_visible:
                                    await self.obs_client.set_marker_enabled(
                                        active_scene, self.config.marker.source_name, True
                                    )
                                    self._marker_visible = True
                            else:
                                if self._marker_visible:
                                    await self.obs_client.set_marker_enabled(
                                        active_scene, self.config.marker.source_name, False
                                    )
                                    self._marker_visible = False
                        else:
                            if self._marker_visible:
                                await self.obs_client.set_marker_enabled(
                                    active_scene, self.config.marker.source_name, False
                                )
                                self._marker_visible = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in CV processing loop: {e}", exc_info=True)

            elapsed = loop.time() - t0
            sleep_time = max(0.001, interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def _obs_scene_sync_loop(self) -> None:
        """Periodically polls OBS for the currently active program scene to detect director manual switches."""
        while self._running:
            try:
                await asyncio.sleep(self.config.switching.scene_poll_interval_sec)
                if self.obs_client.is_connected:
                    active_scene = await self.obs_client.get_current_scene()
                    if active_scene:
                        self.arbiter.update_current_scene(active_scene)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error in OBS scene sync poll: {e}")

    async def _obs_health_loop(self) -> None:
        """Monitors OBS WebSocket connection health and auto-reconnects."""
        reconnect_interval = self.config.obs.reconnect_interval_sec
        while self._running:
            try:
                await asyncio.sleep(reconnect_interval)
                if not self.obs_client.is_connected:
                    logger.debug("OBS disconnected. Attempting auto-reconnection...")
                    success = await self.obs_client.connect()
                    if success:
                        required_scenes = [c.obs_scene for c in self.config.cameras] + [
                            self.config.switching.fallback_scene
                        ]
                        ok, missing = await self.obs_client.verify_scenes(required_scenes)
                        if not ok:
                            available = await self.obs_client.get_scenes()
                            logger.error(
                                f"Connected to OBS Studio, but required scenes are missing: {missing}. "
                                f"Available scenes in OBS: {available}. "
                                "Please add missing scenes to OBS Studio."
                            )
                            continue
                        active_scene = await self.obs_client.get_current_scene()
                        if active_scene:
                            self.arbiter.update_current_scene(active_scene)
                        if self.config.overlay.enabled:
                            await self.obs_client.ensure_text_overlay(
                                self.config.overlay.source_name,
                                self.config.switching.fallback_scene,
                            )
                        if self.config.marker.enabled:
                            await self._init_marker_sources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in OBS connection health monitor: {e}")


async def run_engine(config_path: str, log_level: Optional[str] = None) -> None:
    """Async main routine."""
    config = load_config(config_path)
    active_log_level = log_level or config.logging.level
    setup_logging(active_log_level)

    supervisor = EngineSupervisor(config_path, config)

    # Register OS signals for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Signal handlers not implemented on some platforms/event loops (e.g. Windows)
            pass

    await supervisor.start()
    try:
        await stop_event.wait()
    finally:
        await supervisor.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="uwr-cam-engine: Underwater Rugby Automated Camera Switcher Engine"
    )
    parser.add_argument(
        "-c", "--config",
        dest="config_path",
        default=find_default_config(),
        help="Path to YAML configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override console log level",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run_engine(args.config_path, args.log_level))
    except KeyboardInterrupt:
        pass
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in uwr-cam-engine: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
