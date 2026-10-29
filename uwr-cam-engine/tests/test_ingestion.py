"""Unit tests for low-latency FrameGrabber."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion import FrameGrabber


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
