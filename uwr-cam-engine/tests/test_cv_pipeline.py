"""Unit tests for MotionDetectorMOG2, BallDetectorYOLO, and PipelineManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import BallDetectorConfig, CameraConfig, CameraWeights
from src.cv_pipeline import BallDetectorYOLO, MotionDetectorMOG2
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


def test_ball_detector_disabled():
    detector = BallDetectorYOLO(enabled=False)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    conf, bbox = detector.process_frame(frame)
    assert conf == 0.0
    assert bbox is None


def test_ball_detector_invalid_input():
    detector = BallDetectorYOLO(enabled=False)
    assert detector.process_frame(None) == (0.0, None)
    assert detector.process_frame(np.array([])) == (0.0, None)


def test_ball_detector_mocked_inference():
    detector = BallDetectorYOLO(enabled=False)
    detector.enabled = True
    detector.confidence_threshold = 0.2

    # Mock the internal YOLO model
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=lambda: 0.85)]
    mock_box.xywh = [MagicMock(tolist=lambda: [100, 150, 40, 40])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector._model = mock_model

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    conf, bbox = detector.process_frame(frame)

    assert conf == 0.85
    assert bbox == (100, 150, 40, 40)


def test_pipeline_manager_cv_result_contract():
    configs = [
        CameraConfig(
            id="cam1",
            name="Camera 1",
            rtsp_url="rtsp://localhost:8554/cam1/low",
            obs_scene="cam1",
            weights=CameraWeights(ball=0.7, motion=0.3),
        ),
        CameraConfig(
            id="cam2",
            name="Camera 2",
            rtsp_url="rtsp://localhost:8554/cam2/low",
            obs_scene="cam2",
            weights=CameraWeights(ball=0.7, motion=0.3),
        ),
    ]

    ball_cfg = BallDetectorConfig(enabled=False)
    mgr = PipelineManager(configs, ball_detector_config=ball_cfg)
    results = mgr.process_tick()

    assert "cam1" in results
    assert "cam2" in results

    res1 = results["cam1"]
    assert isinstance(res1, CVResult)
    assert res1.camera_id == "cam1"
    assert res1.motion_score == 0.0
    assert res1.ball_confidence == 0.0
    assert res1.ball_bbox is None
    assert res1.score == 0.0
