"""FFmpeg pipeline builder and camera worker process management."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from typing import List, Optional

from .config import CameraConfig, DefaultsConfig, ServerConfig

logger = logging.getLogger("cam-sim.pipeline")


def build_scale_filter(width: int, height: int, fps: int) -> str:
    """Build an aspect-ratio-preserving scale and letterbox filter."""
    return (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}"
    )


def build_x264_params(gop_size: int) -> str:
    """Build x264 parameters for ultra-low latency and fast stream lock-on.
    
    - keyint & min-keyint: forces strict I-frame/IDR cadence.
    - scenecut=0: disables dynamic scenecut keyframe postponement.
    - repeat-headers=1: repeats SPS and PPS parameter sets with every single I-frame
      so late-joining clients (VLC, OBS) acquire the stream instantaneously.
    """
    return f"keyint={gop_size}:min-keyint={gop_size}:scenecut=0:repeat-headers=1"


def build_ffmpeg_camera_command(
    camera: CameraConfig,
    server: ServerConfig,
    defaults: DefaultsConfig,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Generate the FFmpeg command line arguments for a camera."""
    cmd: List[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "warning",
    ]

    # Real-time pacing and infinite looping
    if defaults.loop:
        cmd.extend(["-stream_loop", "-1"])
    cmd.extend(["-re", "-i", camera.source])

    low = camera.feeds.low_res
    high = camera.feeds.high_res
    x264_params = build_x264_params(defaults.gop_size)

    # Determine active feeds
    if low.enabled and high.enabled:
        # Split filtergraph for dual outputs
        filter_complex = (
            f"[0:v]split=2[v_high_in][v_low_in];"
            f"[v_low_in]{build_scale_filter(low.width, low.height, low.fps)}[v_low_out];"
            f"[v_high_in]{build_scale_filter(high.width, high.height, high.fps)}[v_high_out]"
        )
        cmd.extend(["-filter_complex", filter_complex])

        # Output 1: Low-res feed
        rtsp_low_url = f"rtsp://127.0.0.1:{server.rtsp_port}{low.path}"
        cmd.extend([
            "-map", "[v_low_out]",
            "-c:v", defaults.video_codec,
            "-preset", defaults.preset,
            "-tune", defaults.tune,
            "-g", str(defaults.gop_size),
            "-x264-params", x264_params,
            "-pix_fmt", "yuv420p",
            "-b:v", low.bitrate,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_low_url,
        ])

        # Output 2: High-res feed
        rtsp_high_url = f"rtsp://127.0.0.1:{server.rtsp_port}{high.path}"
        cmd.extend([
            "-map", "[v_high_out]",
            "-c:v", defaults.video_codec,
            "-preset", defaults.preset,
            "-tune", defaults.tune,
            "-g", str(defaults.gop_size),
            "-x264-params", x264_params,
            "-pix_fmt", "yuv420p",
            "-b:v", high.bitrate,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_high_url,
        ])

    elif low.enabled:
        # Only low-res feed enabled
        cmd.extend(["-vf", build_scale_filter(low.width, low.height, low.fps)])
        rtsp_low_url = f"rtsp://127.0.0.1:{server.rtsp_port}{low.path}"
        cmd.extend([
            "-c:v", defaults.video_codec,
            "-preset", defaults.preset,
            "-tune", defaults.tune,
            "-g", str(defaults.gop_size),
            "-x264-params", x264_params,
            "-pix_fmt", "yuv420p",
            "-b:v", low.bitrate,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_low_url,
        ])

    elif high.enabled:
        # Only high-res feed enabled
        cmd.extend(["-vf", build_scale_filter(high.width, high.height, high.fps)])
        rtsp_high_url = f"rtsp://127.0.0.1:{server.rtsp_port}{high.path}"
        cmd.extend([
            "-c:v", defaults.video_codec,
            "-preset", defaults.preset,
            "-tune", defaults.tune,
            "-g", str(defaults.gop_size),
            "-x264-params", x264_params,
            "-pix_fmt", "yuv420p",
            "-b:v", high.bitrate,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            rtsp_high_url,
        ])
    else:
        raise ValueError(f"Camera '{camera.id}' has no enabled feeds")

    return cmd


class CameraWorker:
    """Manages the subprocess execution of an FFmpeg pipeline for a camera."""

    def __init__(
        self,
        camera: CameraConfig,
        server: ServerConfig,
        defaults: DefaultsConfig,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.camera = camera
        self.server = server
        self.defaults = defaults
        self.ffmpeg_bin = ffmpeg_bin
        self.process: Optional[subprocess.Popen] = None
        self._log_thread: Optional[threading.Thread] = None
        self._stopped = False

    def start(self) -> None:
        """Start the FFmpeg worker process."""
        self._stopped = False
        cmd = build_ffmpeg_camera_command(
            self.camera, self.server, self.defaults, self.ffmpeg_bin
        )
        logger.info(f"Starting worker for camera '{self.camera.id}' ({self.camera.name})")
        logger.debug(f"[{self.camera.id}] Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        self._log_thread = threading.Thread(
            target=self._read_stderr,
            name=f"log-{self.camera.id}",
            daemon=True,
        )
        self._log_thread.start()

    def _read_stderr(self) -> None:
        """Stream stderr output from FFmpeg to log warnings/errors."""
        if not self.process or not self.process.stderr:
            return
        for line in self.process.stderr:
            line_str = line.strip()
            if line_str and not self._stopped:
                logger.warning(f"[{self.camera.id}] {line_str}")

    def poll(self) -> Optional[int]:
        """Check if process is still running; returns exit code if terminated."""
        if self.process is None:
            return -1
        return self.process.poll()

    def is_alive(self) -> bool:
        """Return True if worker process is currently running."""
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the FFmpeg worker cleanly."""
        self._stopped = True
        if not self.process:
            return

        logger.info(f"Stopping worker for camera '{self.camera.id}'")
        try:
            if hasattr(os, "killpg") and self.process.pid:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()

            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Worker '{self.camera.id}' did not stop in {timeout}s; killing")
            try:
                if hasattr(os, "killpg") and self.process.pid:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait(timeout=2.0)
            except ProcessLookupError:
                pass
        finally:
            self.process = None
