"""Threaded low-latency RTSP frame grabber with automatic frame dropping."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameGrabber:
    """Ingests video from an RTSP stream in a background thread.

    Bypasses OpenCV's internal FIFO buffer by aggressively calling `grab()`
    and only decoding (`retrieve()`) when requested by the CV pipeline.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        reconnect_interval_sec: float = 3.0,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.reconnect_interval_sec = reconnect_interval_sec

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._last_frame_time: float = 0.0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def start(self) -> None:
        """Starts the background frame grabber thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"grabber-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[{self.camera_id}] Frame grabber thread started for {self.rtsp_url}")

    def stop(self) -> None:
        """Stops the grabber thread and releases the video stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release_cap()
        logger.info(f"[{self.camera_id}] Frame grabber stopped")

    def read_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the latest available frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (Success flag, Latest frame array)
        """
        with self._lock:
            if self._latest_frame is not None:
                # Return a copy or reference depending on processing needs.
                # A shallow copy/reference is sufficient as downstream CV can work directly on it.
                return True, self._latest_frame
            return False, None

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.debug(f"[{self.camera_id}] Error releasing capture: {e}")
            self._cap = None
        with self._lock:
            self._connected = False

    def _init_capture(self) -> bool:
        self._release_cap()
        logger.debug(f"[{self.camera_id}] Opening RTSP stream: {self.rtsp_url}")
        
        # In OpenCV, cv2.CAP_FFMPEG is used. Setting buffer size to 1 if supported.
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(f"[{self.camera_id}] Failed to open RTSP stream: {self.rtsp_url}")
            return False

        self._cap = cap
        with self._lock:
            self._connected = True
        logger.info(f"[{self.camera_id}] Successfully opened RTSP stream")
        return True

    def _capture_loop(self) -> None:
        """Continuous grab loop running on the dedicated background thread."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._init_capture():
                    # Sleep before retrying to prevent busy loop during connection drop
                    time.sleep(self.reconnect_interval_sec)
                    continue

            # grab() blocks on network I/O waiting for the packet, naturally pacing the loop
            grabbed = self._cap.grab()
            if not grabbed:
                logger.warning(f"[{self.camera_id}] Stream disconnected or frame grab failed. Retrying...")
                self._release_cap()
                time.sleep(self.reconnect_interval_sec)
                continue

            # Decode the grabbed frame
            ret, frame = self._cap.retrieve()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
            else:
                # Guard against spin on retrieve error
                time.sleep(0.005)
