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

from .config import CamSimConfig
from .pipeline import CameraWorker

logger = logging.getLogger("cam-sim.supervisor")


class Supervisor:
    """Supervises MediaMTX RTSP server and camera streaming pipelines."""

    def __init__(
        self,
        config: CamSimConfig,
        mediamtx_bin: str = "mediamtx",
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.config = config
        self.mediamtx_bin = mediamtx_bin
        self.ffmpeg_bin = ffmpeg_bin
        self.mediamtx_proc: Optional[subprocess.Popen] = None
        self.workers: Dict[str, CameraWorker] = {}
        self._running = False
        self._stop_event = threading.Event()

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

    def start_camera_workers(self) -> None:
        """Initialize and start FFmpeg workers for all cameras."""
        for camera in self.config.cameras:
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

    def stop_camera_workers(self) -> None:
        """Stop all running camera workers."""
        for cam_id, worker in self.workers.items():
            worker.stop()
        self.workers.clear()

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

            # Health monitoring loop
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=2.0)
                if self._stop_event.is_set():
                    break

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
