"""Unit tests for low-latency FrameGrabber."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion import FrameGrabber, get_decoding_core_pool


def test_grabber_initial_state():
    grabber = FrameGrabber(camera_id="cam1", rtsp_url="rtsp://localhost:8554/cam1/low")
    assert not grabber.is_connected
    ret, frame = grabber.read_latest()
    assert ret is False
    assert frame is None


def test_grabber_start_stop(mocker):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.grab.return_value = True
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    mock_cap.retrieve.return_value = (True, dummy_frame)

    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    grabber = FrameGrabber(camera_id="cam1", rtsp_url="rtsp://localhost:8554/cam1/low")
    grabber.start()

    # Allow thread to start and capture at least one frame
    time.sleep(0.1)

    assert grabber.is_connected
    ret, frame = grabber.read_latest()
    assert ret is True
    assert frame is not None
    assert frame.shape == (360, 640, 3)

    grabber.stop()
    assert not grabber.is_connected
    mock_cap.release.assert_called_once()


def test_grabber_reconnect_on_grab_failure(mocker):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # grab fails
    mock_cap.grab.return_value = False

    mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    grabber = FrameGrabber(
        camera_id="cam1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        reconnect_interval_sec=0.01,
    )
    grabber.start()
    time.sleep(0.05)

    grabber.stop()
    assert mock_cap.release.called


def test_get_decoding_core_pool_apple_silicon(mocker):
    """Verify Apple Silicon isolates Performance cores (perflevel1)."""
    mocker.patch("sys.platform", "darwin")
    mocker.patch("subprocess.check_output", return_value="12\n")

    pool = get_decoding_core_pool()
    assert pool == 12


def test_get_decoding_core_pool_intel_mac(mocker):
    """Verify Intel Mac falls back to hw.physicalcpu when perflevel1 does not exist."""
    mocker.patch("sys.platform", "darwin")

    def mock_check_output(cmd, **kwargs):
        if "hw.perflevel1.physicalcpu" in cmd:
            raise Exception("sysctl: unknown oid 'hw.perflevel1.physicalcpu'")
        if "hw.physicalcpu" in cmd:
            return "4\n"
        raise Exception("unknown sysctl")

    mocker.patch("subprocess.check_output", side_effect=mock_check_output)

    pool = get_decoding_core_pool()
    assert pool == 4


def test_get_decoding_core_pool_linux_fallback(mocker):
    """Verify Linux and containers fall back to os.cpu_count()."""
    mocker.patch("sys.platform", "linux")
    mocker.patch("os.cpu_count", return_value=8)

    pool = get_decoding_core_pool()
    assert pool == 8


def test_grabber_passes_n_threads_param(mocker):
    """Verify FrameGrabber passes CAP_PROP_N_THREADS when num_threads is specified."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_vc = mocker.patch("cv2.VideoCapture", return_value=mock_cap)

    import cv2

    grabber = FrameGrabber(
        camera_id="cam1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        num_threads=2,
    )
    success = grabber._init_capture()

    assert success is True
    assert grabber.is_connected is True
    assert grabber.connection_epoch == 1

    # Verify VideoCapture called with [CAP_PROP_N_THREADS, 2]
    mock_vc.assert_called_once_with(
        "rtsp://localhost:8554/cam1/low",
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_N_THREADS, 2],
    )
    mock_cap.set.assert_called_once_with(cv2.CAP_PROP_BUFFERSIZE, 1)

