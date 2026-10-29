"""Unit tests for MotionDetectorMOG2 and PipelineManager."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CameraConfig
from src.cv_pipeline import MotionDetectorMOG2
from src.pipeline_manager import CVResult, PipelineManager


def test_motion_detector_blank_frames():
    detector = MotionDetectorMOG2(history=10, min_contour_area=50)
    blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # First few frames establish background model
    for _ in range(5):
        score = detector.process_frame(blank_frame)

    # Static frame should have 0.0 motion
    score = detector.process_frame(blank_frame)
    assert score == 0.0


def test_motion_detector_detects_significant_motion():
    detector = MotionDetectorMOG2(history=20, min_contour_area=50)
    bg_frame = np.ones((360, 640, 3), dtype=np.uint8) * 100

    # Train background model on static frame
    for _ in range(15):
        detector.process_frame(bg_frame)

    # Introduce a large moving object (e.g. 100x100 white square)
    motion_frame = bg_frame.copy()
    motion_frame[50:200, 50:200] = 255  # 150x150 block

    score = detector.process_frame(motion_frame)
    assert score > 0.0
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_motion_detector_invalid_input():
    detector = MotionDetectorMOG2()
    assert detector.process_frame(None) == 0.0
    assert detector.process_frame(np.array([])) == 0.0


def test_pipeline_manager_cv_result_contract():
    configs = [
        CameraConfig(
            id="cam1",
            name="Camera 1",
            rtsp_url="rtsp://localhost:8554/cam1/low",
            obs_scene="cam1",
        ),
        CameraConfig(
            id="cam2",
            name="Camera 2",
            rtsp_url="rtsp://localhost:8554/cam2/low",
            obs_scene="cam2",
        ),
    ]

    mgr = PipelineManager(configs)
    results = mgr.process_tick()

    assert "cam1" in results
    assert "cam2" in results

    res1 = results["cam1"]
    assert isinstance(res1, CVResult)
    assert res1.camera_id == "cam1"
    assert res1.motion_score == 0.0  # Not running yet
    assert res1.ball_confidence == 0.0
    assert res1.ball_bbox is None
