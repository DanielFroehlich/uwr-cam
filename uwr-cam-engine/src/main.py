"""CLI entrypoint and central orchestration loop for uwr-cam-engine."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .config import EngineConfig, load_config
from .config_watcher import ConfigWatcher
from .obs import OBSClientProtocol, ObswsPythonClient
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
        self.pipeline_mgr: PipelineManager = PipelineManager(self.config.cameras)
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._cv_task: Optional[asyncio.Task] = None
        self._last_status_log_time: float = 0.0

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
            await self.obs_client.verify_scenes(required_scenes)
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
        logger.info("Applying dynamic configuration updates...")
        self.config = new_cfg

        # Check log level change
        if old_cfg.logging.level != new_cfg.logging.level:
            setup_logging(new_cfg.logging.level)
            logger.info(f"Updated log level to {new_cfg.logging.level}")

        # Check OBS connection configuration change
        if old_cfg.obs != new_cfg.obs:
            logger.info("OBS connection parameters changed. Reconnecting OBS client...")
            await self.obs_client.disconnect()
            self.obs_client = ObswsPythonClient(new_cfg.obs)
            await self.obs_client.connect()

        # Re-verify required scenes with new camera list / fallback scene
        required_scenes = [c.obs_scene for c in new_cfg.cameras] + [
            new_cfg.switching.fallback_scene
        ]
        await self.obs_client.verify_scenes(required_scenes)

        # Check if cameras changed: restart pipelines if needed
        if old_cfg.cameras != new_cfg.cameras:
            logger.info("Camera configuration updated. Restarting video ingestion pipelines...")
            self.pipeline_mgr.stop_all()
            self.pipeline_mgr = PipelineManager(new_cfg.cameras)
            self.pipeline_mgr.start_all()

        logger.info(
            f"Configuration reloaded: {len(new_cfg.cameras)} cameras active, "
            f"fallback scene: '{new_cfg.switching.fallback_scene}'."
        )

    async def _cv_processing_loop(self) -> None:
        """Main async loop ticking at ~30 FPS to process CV and report status."""
        loop = asyncio.get_running_loop()
        interval = 1.0 / 30.0  # Target ~30 FPS
        self._last_status_log_time = loop.time()

        while self._running:
            t0 = loop.time()
            try:
                # Run the CV frame processing in threadpool to avoid blocking event loop
                results = await asyncio.to_thread(self.pipeline_mgr.process_tick)
                
                # Granular debug logging
                logger.debug(f"CV Tick results: {results}")

                # Periodic summary logging at INFO level
                now = loop.time()
                status_interval = self.config.logging.status_interval_sec
                if now - self._last_status_log_time >= status_interval:
                    self._last_status_log_time = now
                    summary_parts = [
                        f"{cam_id}: [motion={res.motion_score:.2f}]"
                        for cam_id, res in results.items()
                    ]
                    active_scene = await self.obs_client.get_current_scene() or "unknown"
                    logger.info(
                        f"[Status] Active Scene: '{active_scene}' | CV Scores: "
                        + ", ".join(summary_parts)
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in CV processing loop: {e}", exc_info=True)

            elapsed = loop.time() - t0
            sleep_time = max(0.001, interval - elapsed)
            await asyncio.sleep(sleep_time)

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
                        await self.obs_client.verify_scenes(required_scenes)
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
    except Exception as e:
        logger.error(f"Fatal error in uwr-cam-engine: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
