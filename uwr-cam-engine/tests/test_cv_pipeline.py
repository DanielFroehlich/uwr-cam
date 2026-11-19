"""Unit tests for MotionDetectorMOG2, BallDetectorYOLO, and PipelineManager."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import (
    BallDetectorConfig,
    CameraConfig,
    CameraWeights,
    MotionDetectorConfig,
)
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
    assert detector.last_inference_time_ms >= 0.0


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
    assert res1.inference_time_ms == 0.0


def test_camera_pipeline_measures_inference_latency(mocker):
    """Verify that CameraPipeline measures ball detector latency and includes it in CVResult."""
    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
        weights=CameraWeights(ball=0.7, motion=0.3),
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.process_frame.return_value = (0.9, (10, 20, 30, 40))

    from src.pipeline_manager import CameraPipeline

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)
    # Mock grabber returning a frame
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, dummy_frame))

    result = pipe.process_latest()

    assert result.camera_id == "cam1"
    assert result.ball_confidence == 0.9
    assert result.inference_time_ms >= 0.0
    assert mock_detector.process_frame.called


def test_camera_pipeline_measures_per_stage_latencies_and_total(mocker):
    """Verify CameraPipeline individually measures transform, motion, and ball detection latencies, reporting total."""
    from src.pipeline_manager import CameraPipeline

    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.target_resolution = (640, 360)
    mock_detector.process_frame.return_value = (0.85, (100, 100, 30, 30))

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)
    # Stream sends 704x576 requiring transform
    stream_frame = np.zeros((576, 704, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, stream_frame))

    res = pipe.process_latest()

    # Individual stage times must be recorded
    assert res.transform_time_ms >= 0.0
    assert res.motion_time_ms >= 0.0
    assert res.ball_time_ms >= 0.0
    # Total time equals sum of stages
    expected_total = round(res.transform_time_ms + res.motion_time_ms + res.ball_time_ms, 2)
    assert res.total_time_ms == expected_total
    # inference_time_ms reports the total time
    assert res.inference_time_ms == res.total_time_ms



def test_ball_detector_yolo_target_resolution_derivation(mocker):
    """Verify that BallDetectorYOLO derives target resolution from model native imgsz and aspect ratio."""
    mocker.patch("src.cv_pipeline.BallDetectorYOLO._load_model")
    detector = BallDetectorYOLO(enabled=False, aspect_ratio=(16, 9))
    detector.native_imgsz = 640
    detector._compute_target_resolution()
    assert detector.target_resolution == (640, 360)

    # 4:3 Aspect ratio
    detector.aspect_ratio = (4, 3)
    detector._compute_target_resolution()
    assert detector.target_resolution == (640, 480)

    # None aspect ratio defaults to square native imgsz
    detector.aspect_ratio = None
    detector._compute_target_resolution()
    assert detector.target_resolution == (640, 640)

    # Dynamic update
    detector.update_config(enabled=False, aspect_ratio=(16, 9))
    assert detector.target_resolution == (640, 360)


def test_camera_pipeline_normalizes_704x576_to_model_resolution(mocker):
    """Verify that CameraPipeline automatically normalizes 704x576 anamorphic stream to model target 640x360."""
    from src.pipeline_manager import CameraPipeline

    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.target_resolution = (640, 360)
    mock_detector.process_frame.return_value = (0.85, (320, 180, 50, 50))

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)
    # Stream sends 704x576
    stream_frame = np.zeros((576, 704, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, stream_frame))

    res = pipe.process_latest()

    # Output CVResult should reflect normalized 640x360 dimensions
    assert res.frame_width == 640
    assert res.frame_height == 360
    assert pipe._resize_target == (640, 360)

    # Verify detector received frame resized to (360, 640, 3)
    call_frame = mock_detector.process_frame.call_args[0][0]
    assert call_frame.shape == (360, 640, 3)


def test_camera_pipeline_normalizes_1080p_to_model_resolution(mocker):
    """Verify that CameraPipeline downsamples 1080p streams to 640x360 for efficient CV processing."""
    from src.pipeline_manager import CameraPipeline

    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.target_resolution = (640, 360)
    mock_detector.process_frame.return_value = (0.0, None)

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)
    hd_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, hd_frame))

    res = pipe.process_latest()

    assert res.frame_width == 640
    assert res.frame_height == 360
    assert pipe._resize_target == (640, 360)

    call_frame = mock_detector.process_frame.call_args[0][0]
    assert call_frame.shape == (360, 640, 3)


def test_camera_pipeline_passthrough_when_stream_matches_target_resolution(mocker):
    """Verify that CameraPipeline performs zero-copy pass-through when stream already matches model resolution."""
    from src.pipeline_manager import CameraPipeline

    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.target_resolution = (640, 360)
    mock_detector.process_frame.return_value = (0.0, None)

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)
    native_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, native_frame))
    mock_resize = mocker.patch("cv2.resize")

    res = pipe.process_latest()

    assert res.frame_width == 640
    assert res.frame_height == 360
    assert pipe._resize_target is None
    # cv2.resize must NOT have been called
    mock_resize.assert_not_called()


def test_camera_pipeline_re_evaluates_on_stream_reconnect(mocker):
    """Verify that CameraPipeline re-evaluates transformation when grabber connection epoch changes."""
    from src.pipeline_manager import CameraPipeline

    cfg = CameraConfig(
        id="cam1",
        name="Camera 1",
        rtsp_url="rtsp://localhost:8554/cam1/low",
        obs_scene="cam1",
    )
    mock_detector = MagicMock()
    mock_detector.enabled = True
    mock_detector.target_resolution = (640, 360)
    mock_detector.process_frame.return_value = (0.0, None)

    pipe = CameraPipeline(cfg, ball_detector=mock_detector)

    # Initial session (epoch 1): stream is 640x360
    pipe.grabber._connection_epoch = 1
    frame_epoch1 = np.zeros((360, 640, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, frame_epoch1))
    pipe.process_latest()
    assert pipe._resize_target is None
    assert pipe._last_evaluated_epoch == 1

    # Reconnection event (epoch 2): camera reconnects with 704x576 stream
    pipe.grabber._connection_epoch = 2
    frame_epoch2 = np.zeros((576, 704, 3), dtype=np.uint8)
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, frame_epoch2))
    pipe.process_latest()
    assert pipe._resize_target == (640, 360)
    assert pipe._last_evaluated_epoch == 2


def test_pipeline_manager_hot_reload_updates_aspect_ratio_and_invalidates(mocker):
    """Verify that PipelineManager updates aspect ratio and invalidates pipeline transforms on reload."""
    mocker.patch("src.cv_pipeline.BallDetectorYOLO._load_model")
    cam_cfg = [
        CameraConfig(
            id="cam1",
            name="Camera 1",
            rtsp_url="rtsp://localhost:8554/cam1/low",
            obs_scene="cam1",
        )
    ]
    det_cfg = BallDetectorConfig(enabled=False, aspect_ratio=(16, 9))
    mgr = PipelineManager(cam_cfg, ball_detector_config=det_cfg)
    mgr.shared_ball_detector.native_imgsz = 640
    mgr.shared_ball_detector._compute_target_resolution()
    assert mgr.shared_ball_detector.target_resolution == (640, 360)

    # Initial tick evaluates transform
    pipe = mgr.pipelines["cam1"]
    mocker.patch.object(pipe.grabber, "read_latest", return_value=(True, np.zeros((576, 704, 3), dtype=np.uint8)))
    pipe.process_latest()
    assert pipe._resize_target == (640, 360)

    # Hot reload with new 4:3 aspect ratio
    new_det_cfg = BallDetectorConfig(enabled=False, aspect_ratio=(4, 3))
    mgr.update_ball_detector(new_det_cfg)
    assert mgr.shared_ball_detector.target_resolution == (640, 480)
    assert pipe._last_evaluated_epoch == -1  # Invalidated

    # Next tick re-evaluates transform to (640, 480)
    pipe.process_latest()
    assert pipe._resize_target == (640, 480)


def test_pipeline_manager_allocates_core_aware_threads(mocker):
    """Verify PipelineManager allocates threads based on hardware core pool divided by camera count."""
    mocker.patch("src.cv_pipeline.BallDetectorYOLO._load_model")
    # Simulate 12 Performance cores on M5 Pro
    mocker.patch("src.pipeline_manager.get_decoding_core_pool", return_value=12)

    cam_configs = [
        CameraConfig(id=f"cam{i}", name=f"Cam {i}", rtsp_url=f"rtsp://localhost/cam{i}", obs_scene=f"cam{i}")
        for i in range(1, 4)  # 3 cameras
    ]
    det_cfg = BallDetectorConfig(enabled=False)
    mgr = PipelineManager(cam_configs, ball_detector_config=det_cfg)

    # 12 cores // 3 cameras = 4 threads per camera (capped at 4)
    for pipe in mgr.pipelines.values():
        assert pipe.grabber.num_threads == 4


def test_pipeline_manager_motion_detector_config_and_update(mocker):
    """Verify PipelineManager initializes and dynamically updates motion_detector min_contour_area."""
    mocker.patch("src.cv_pipeline.BallDetectorYOLO._load_model")
    cam_configs = [
        CameraConfig(id="cam1", name="Cam 1", rtsp_url="rtsp://localhost/cam1", obs_scene="cam1"),
        CameraConfig(id="cam2", name="Cam 2", rtsp_url="rtsp://localhost/cam2", obs_scene="cam2"),
    ]
    motion_cfg = MotionDetectorConfig(min_contour_area=200)
    mgr = PipelineManager(
        cam_configs,
        ball_detector_config=BallDetectorConfig(enabled=False),
        motion_detector_config=motion_cfg,
    )

    # Initial value
    for pipe in mgr.pipelines.values():
        assert pipe.motion_detector.min_contour_area == 200

    # Dynamic hot-reload update
    new_motion_cfg = MotionDetectorConfig(min_contour_area=500)
    mgr.update_motion_detector(new_motion_cfg)

    for pipe in mgr.pipelines.values():
        assert pipe.motion_detector.min_contour_area == 500



