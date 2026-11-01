"""Supervisor managing MediaMTX RTSP server and FFmpeg camera workers."""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import CameraConfig, CamSimConfig, load_config
from .pipeline import CameraWorker

logger = logging.getLogger("cam-sim.supervisor")


class Supervisor:
    """Supervises MediaMTX RTSP server and camera streaming pipelines."""

    def __init__(
        self,
        config: CamSimConfig,
        config_path: Optional[str | Path] = None,
        mediamtx_bin: str = "mediamtx",
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.config = config
        self.config_path: Optional[Path] = Path(config_path).resolve() if config_path else None
        self.mediamtx_bin = mediamtx_bin
        self.ffmpeg_bin = ffmpeg_bin
        self.mediamtx_proc: Optional[subprocess.Popen] = None
        self.workers: Dict[str, CameraWorker] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._last_mtime: Optional[int] = self._get_mtime()

    def _get_mtime(self) -> Optional[int]:
        """Safely fetch config file modification time."""
        if not self.config_path:
            return None
        try:
            return self.config_path.stat().st_mtime_ns
        except OSError:
            return None

    def _wait_for_port(self, host: str, port: int, timeout: float = 10.0) -> bool:
        """Wait until a TCP port is open and accepting connections."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.2)
        return False

    def start_mediamtx(self) -> None:
        """Launch the embedded MediaMTX RTSP server."""
        logger.info(f"Starting MediaMTX RTSP server on port {self.config.server.rtsp_port}...")

        # Write MediaMTX configuration allowing all dynamic camera stream paths
        mediamtx_conf = Path("/tmp/mediamtx.yml")
        transports_str = ", ".join(self.config.server.rtsp_transports)
        conf_content = (
            f"rtspAddress: '{self.config.server.bind_address}:{self.config.server.rtsp_port}'\n"
            f"rtspTransports: [{transports_str}]\n"
            f"logLevel: warn\n"
            f"paths:\n"
            f"  all_others:\n"
        )
        mediamtx_conf.write_text(conf_content)

        try:
            self.mediamtx_proc = subprocess.Popen(
                [self.mediamtx_bin, str(mediamtx_conf)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"MediaMTX binary '{self.mediamtx_bin}' not found. "
                "Ensure MediaMTX is installed or run within the container."
            )

        # Wait for RTSP port to become ready
        if not self._wait_for_port("127.0.0.1", self.config.server.rtsp_port, timeout=8.0):
            if self.mediamtx_proc.poll() is not None:
                _, stderr = self.mediamtx_proc.communicate()
                raise RuntimeError(f"MediaMTX exited prematurely: {stderr}")
            raise RuntimeError(f"Timed out waiting for MediaMTX on port {self.config.server.rtsp_port}")

        logger.info(f"MediaMTX RTSP server is active on port {self.config.server.rtsp_port}")

    def stop_mediamtx(self) -> None:
        """Gracefully terminate MediaMTX."""
        if not self.mediamtx_proc:
            return

        logger.info("Stopping MediaMTX RTSP server...")
        try:
            if hasattr(os, "killpg") and self.mediamtx_proc.pid:
                try:
                    os.killpg(os.getpgid(self.mediamtx_proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.mediamtx_proc.terminate()

            self.mediamtx_proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg") and self.mediamtx_proc.pid:
                try:
                    os.killpg(os.getpgid(self.mediamtx_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.mediamtx_proc.kill()
        finally:
            self.mediamtx_proc = None

    def _log_active_streams(self) -> None:
        """Log active RTSP streams and camera endpoints."""
        logger.info("=" * 60)
        logger.info(f"cam-sim is live! Simulating {len(self.config.cameras)} cameras.")
        for cam in self.config.cameras:
            low = cam.feeds.low_res
            high = cam.feeds.high_res
            port = self.config.server.rtsp_port
            logger.info(f"  * Camera '{cam.id}' ({cam.name}):")
            if low.enabled:
                logger.info(f"      Low-Res:  rtsp://<host>:{port}{low.path} ({low.width}x{low.height} @ {low.fps}fps)")
            if high.enabled:
                logger.info(f"      High-Res: rtsp://<host>:{port}{high.path} ({high.width}x{high.height} @ {high.fps}fps)")
        logger.info("=" * 60)

    def _start_camera_worker(self, camera: CameraConfig) -> None:
        """Initialize and start an FFmpeg worker for a single camera."""
        source_path = Path(camera.source)
        if not source_path.exists():
            logger.warning(
                f"Camera '{camera.id}' source file does not exist: {source_path}. "
                "Worker will still attempt to launch or wait for file."
            )

        worker = CameraWorker(
            camera=camera,
            server=self.config.server,
            defaults=self.config.defaults,
            ffmpeg_bin=self.ffmpeg_bin,
        )
        self.workers[camera.id] = worker
        try:
            worker.start()
        except Exception as e:
            logger.error(f"Failed to start worker for camera '{camera.id}': {e}")

    def start_camera_workers(self) -> None:
        """Initialize and start FFmpeg workers for all cameras."""
        for camera in self.config.cameras:
            self._start_camera_worker(camera)

    def stop_camera_workers(self) -> None:
        """Stop all running camera workers."""
        for cam_id, worker in self.workers.items():
            worker.stop()
        self.workers.clear()

    def check_config_updates(self) -> None:
        """Check if configuration file has changed on disk, reload and apply diff."""
        if not self.config_path:
            return

        current_mtime = self._get_mtime()
        if current_mtime is None or current_mtime == self._last_mtime:
            return

        self._last_mtime = current_mtime
        logger.info(f"Detected modification in {self.config_path.name}. Validating new configuration...")

        try:
            new_config = load_config(self.config_path)
        except Exception as e:
            logger.error(
                f"Hot-reload rejected: Configuration validation failed: {e}. Keeping current active configuration."
            )
            return

        if new_config == self.config:
            logger.debug("Config file updated but contents are identical. No action needed.")
            return

        self.apply_config_update(new_config)

    def apply_config_update(self, new_config: CamSimConfig) -> None:
        """Apply dynamic configuration updates."""
        old_config = self.config
        logger.info("New configuration validated successfully. Applying dynamic updates...")

        # 1. MediaMTX RTSP Server settings changed
        if old_config.server != new_config.server:
            logger.info("Server configuration changed. Restarting MediaMTX and all camera workers...")
            self.stop_camera_workers()
            self.stop_mediamtx()
            self.config = new_config
            try:
                self.start_mediamtx()
                self.start_camera_workers()
                self._log_active_streams()
                logger.info("Server and camera workers restarted successfully with new configuration.")
            except Exception as e:
                logger.error(f"Error restarting services after server configuration change: {e}", exc_info=True)
            return

        # 2. Defaults (codec, preset, tune, gop_size, loop) changed
        if old_config.defaults != new_config.defaults:
            logger.info("Default encoding settings changed. Restarting all camera workers...")
            self.stop_camera_workers()
            self.config = new_config
            self.start_camera_workers()
            self._log_active_streams()
            logger.info("All camera workers restarted successfully with new defaults.")
            return

        # 3. Camera definitions changed (surgical worker restarts)
        self.config = new_config
        old_cams = {c.id: c for c in old_config.cameras}
        new_cams = {c.id: c for c in new_config.cameras}

        # Stop workers for removed cameras
        for cam_id in list(self.workers.keys()):
            if cam_id not in new_cams:
                logger.info(f"Camera '{cam_id}' removed in new config. Stopping worker...")
                self.workers[cam_id].stop()
                del self.workers[cam_id]

        # Restart workers for modified cameras
        for cam_id, new_cam in new_cams.items():
            if cam_id in old_cams and old_cams[cam_id] != new_cam:
                logger.info(f"Camera '{cam_id}' configuration changed. Restarting worker...")
                if cam_id in self.workers:
                    self.workers[cam_id].stop()
                    del self.workers[cam_id]
                self._start_camera_worker(new_cam)

        # Start workers for newly added cameras
        for cam_id, new_cam in new_cams.items():
            if cam_id not in old_cams:
                logger.info(f"New camera '{cam_id}' added. Starting worker...")
                self._start_camera_worker(new_cam)

        self._log_active_streams()
        logger.info(f"Camera configuration reloaded successfully: {len(self.workers)} camera workers active.")

    def run(self) -> None:
        """Run the supervisor main loop until interrupted."""
        self._running = True
        self._stop_event.clear()

        # Register OS signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass

        try:
            self.start_mediamtx()
            self.start_camera_workers()
            self._log_active_streams()

            # Health monitoring loop
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=2.0)
                if self._stop_event.is_set():
                    break

                # Check for dynamic configuration updates
                self.check_config_updates()

                # Monitor MediaMTX health
                if self.mediamtx_proc and self.mediamtx_proc.poll() is not None:
                    logger.error("MediaMTX server stopped unexpectedly! Restarting...")
                    self.start_mediamtx()

                # Monitor camera workers
                for cam in self.config.cameras:
                    worker = self.workers.get(cam.id)
                    if worker and not worker.is_alive():
                        exit_code = worker.poll()
                        logger.warning(
                            f"Worker for camera '{cam.id}' exited with code {exit_code}. Restarting..."
                        )
                        time.sleep(1.0)
                        try:
                            worker.start()
                        except Exception as e:
                            logger.error(f"Failed to restart worker for camera '{cam.id}': {e}")

        finally:
            self.shutdown()

    def _handle_signal(self, signum: int, frame) -> None:
        """Signal handler callback."""
        logger.info(f"Received shutdown signal ({signum}), stopping cam-sim...")
        self._stop_event.set()

    def shutdown(self) -> None:
        """Shutdown all supervised services."""
        if not self._running:
            return
        self._running = False
        logger.info("Shutting down cam-sim supervisor...")
        self.stop_camera_workers()
        self.stop_mediamtx()
        logger.info("cam-sim supervisor stopped successfully.")
