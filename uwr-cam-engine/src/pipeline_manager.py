"""Pipeline manager coordinating multi-camera frame grabbing and computer vision analysis."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import BallDetectorConfig, CameraConfig, CameraWeights, MotionDetectorConfig
from .cv_pipeline import (
    BallDetectorProtocol,
    BallDetectorYOLO,
    MotionDetectorMOG2,
    MotionDetectorProtocol,
)
from .ingestion import FrameGrabber, get_decoding_core_pool

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Structured result of computer vision processing for a camera."""
    camera_id: str
    motion_score: float = 0.0          # Phase 3: Motion density [0.0, 1.0]
    ball_confidence: float = 0.0       # Phase 4: Ball detection confidence [0.0, 1.0]
    ball_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    score: float = 0.0                 # Phase 4: Composite score = (w_ball * B) + (w_motion * M)
    frame_width: int = 640
    frame_height: int = 360
    transform_time_ms: float = 0.0     # Time spent on frame scaling / aspect ratio normalization
    motion_time_ms: float = 0.0        # Time spent on MOG2 motion detection
    ball_time_ms: float = 0.0          # Time spent on YOLO ball detector inference
    total_time_ms: float = 0.0         # Total CV processing time (transform + motion + ball)
    inference_time_ms: float = 0.0     # Reported total frame processing latency in milliseconds


class CameraPipeline:
    """Combines a single camera frame grabber with its detection models."""

    def __init__(
        self,
        config: CameraConfig,
        ball_detector: Optional[BallDetectorProtocol] = None,
        motion_detector: Optional[MotionDetectorProtocol] = None,
        min_contour_area: int = 300,
        num_threads: Optional[int] = None,
    ) -> None:
        self.config = config
        self.grabber = FrameGrabber(
            camera_id=config.id,
            rtsp_url=config.rtsp_url,
            num_threads=num_threads,
        )
        self.motion_detector: MotionDetectorProtocol = (
            motion_detector or MotionDetectorMOG2(min_contour_area=min_contour_area)
        )
        self.ball_detector: BallDetectorProtocol = ball_detector or BallDetectorYOLO()

        self._last_evaluated_epoch: int = -1
        self._last_stream_shape: Optional[Tuple[int, int]] = None
        self._resize_target: Optional[Tuple[int, int]] = None

    def invalidate_transform(self) -> None:
        """Reset cached transform to force re-evaluation on next frame."""
        self._last_evaluated_epoch = -1
        self._last_stream_shape = None

    def _evaluate_stream_transform(self, fw: int, fh: int, epoch: int) -> None:
        """Evaluate stream resolution against detector target resolution once per connection."""
        self._last_evaluated_epoch = epoch
        self._last_stream_shape = (fw, fh)

        raw_res = getattr(self.ball_detector, "target_resolution", None)
        target_res: Optional[Tuple[int, int]] = None
        if isinstance(raw_res, (tuple, list)) and len(raw_res) == 2:
            try:
                target_res = (int(raw_res[0]), int(raw_res[1]))
            except (ValueError, TypeError):
                target_res = None

        if target_res is None or (fw, fh) == target_res:
            self._resize_target = None
            logger.info(
                f"[{self.config.id}] Stream resolution {fw}x{fh} matches model target resolution. "
                "Zero-copy passthrough active."
            )
        else:
            self._resize_target = target_res
            stream_ar = fw / max(1, fh)
            target_ar = target_res[0] / max(1, target_res[1])
            logger.info(
                f"[{self.config.id}] Stream connected at {fw}x{fh} (AR {stream_ar:.2f}). "
                f"Configured automatic normalization to {target_res[0]}x{target_res[1]} (AR {target_ar:.2f})."
            )

    def start(self) -> None:
        self.grabber.start()

    def stop(self) -> None:
        self.grabber.stop()

    def process_latest(self) -> CVResult:
        """Pulls the latest frame and executes the CV pipeline."""
        ret, frame = self.grabber.read_latest()
        if not ret or frame is None:
            # Stream not ready or offline
            return CVResult(camera_id=self.config.id, motion_score=0.0, score=0.0)

        # 0. Check if stream transform needs evaluation (connection established, reconnected, or shape changed)
        current_epoch = getattr(self.grabber, "connection_epoch", 0)
        fh, fw = frame.shape[:2]
        if current_epoch != self._last_evaluated_epoch or (fw, fh) != self._last_stream_shape:
            self._evaluate_stream_transform(fw, fh, current_epoch)

        # 1. Apply precomputed transform if active (measure normalization time)
        transform_time_ms = 0.0
        if self._resize_target is not None:
            t_trans = time.perf_counter()
            frame = cv2.resize(frame, self._resize_target, interpolation=cv2.INTER_LINEAR)
            transform_time_ms = (time.perf_counter() - t_trans) * 1000.0

        # 2. Compute motion density (measure motion detection time)
        t_motion = time.perf_counter()
        motion_score = self.motion_detector.process_frame(frame)
        motion_time_ms = (time.perf_counter() - t_motion) * 1000.0

        # 3. Check if ball detection is enabled on detector (measure ball detection time)
        is_ball_enabled = getattr(self.ball_detector, "enabled", True)
        ball_time_ms = 0.0
        if is_ball_enabled:
            t_ball = time.perf_counter()
            ball_conf, ball_bbox = self.ball_detector.process_frame(frame)
            ball_time_ms = (time.perf_counter() - t_ball) * 1000.0
            # Calculate composite camera score: S = (w_ball * B) + (w_motion * M)
            weights = self.config.weights or CameraWeights()
            w_ball = weights.ball
            w_motion = weights.motion
            composite_score = (w_ball * ball_conf) + (w_motion * motion_score)
        else:
            # Ball detector disabled: motion receives 100% weight
            ball_conf = 0.0
            ball_bbox = None
            composite_score = motion_score

        r_transform = round(transform_time_ms, 2)
        r_motion = round(motion_time_ms, 2)
        r_ball = round(ball_time_ms, 2)
        r_total = round(r_transform + r_motion + r_ball, 2)

        out_h, out_w = frame.shape[:2]
        return CVResult(
            camera_id=self.config.id,
            motion_score=motion_score,
            ball_confidence=ball_conf,
            ball_bbox=ball_bbox,
            score=round(float(composite_score), 4),
            frame_width=out_w,
            frame_height=out_h,
            transform_time_ms=r_transform,
            motion_time_ms=r_motion,
            ball_time_ms=r_ball,
            total_time_ms=r_total,
            inference_time_ms=r_total,
        )


class PipelineManager:
    """Manages all active camera pipelines, sharing a single GPU ball detector instance."""

    def __init__(
        self,
        camera_configs: List[CameraConfig],
        ball_detector_config: Optional[BallDetectorConfig] = None,
        motion_detector_config: Optional[MotionDetectorConfig] = None,
        ball_detector: Optional[BallDetectorProtocol] = None,
    ) -> None:
        self.motion_detector_config = motion_detector_config or MotionDetectorConfig()

        if ball_detector is None:
            if ball_detector_config is not None:
                self.shared_ball_detector: BallDetectorProtocol = BallDetectorYOLO(
                    model_name=ball_detector_config.model_name,
                    confidence_threshold=ball_detector_config.confidence_threshold,
                    classes=ball_detector_config.classes,
                    enabled=ball_detector_config.enabled,
                    aspect_ratio=ball_detector_config.aspect_ratio,
                )
            else:
                self.shared_ball_detector = BallDetectorYOLO()
        else:
            self.shared_ball_detector = ball_detector

        core_pool = get_decoding_core_pool()
        num_cams = max(1, len(camera_configs))
        threads_per_camera = max(1, min(4, core_pool // num_cams))
        logger.info(
            f"Hardware topology: {core_pool} decoding cores available. "
            f"Allocating {threads_per_camera} decoder threads per camera across {num_cams} cameras."
        )

        self.pipelines: Dict[str, CameraPipeline] = {
            cam.id: CameraPipeline(
                cam,
                ball_detector=self.shared_ball_detector,
                min_contour_area=self.motion_detector_config.min_contour_area,
                num_threads=threads_per_camera,
            )
            for cam in camera_configs
        }

    def update_motion_detector(self, motion_detector_config: MotionDetectorConfig) -> None:
        """Update motion detector settings dynamically without restarting video streams."""
        self.motion_detector_config = motion_detector_config
        for pipe in self.pipelines.values():
            if hasattr(pipe.motion_detector, "min_contour_area"):
                pipe.motion_detector.min_contour_area = motion_detector_config.min_contour_area
        logger.info(
            f"Motion detector dynamically updated: min_contour_area={motion_detector_config.min_contour_area}"
        )

    def update_ball_detector(self, ball_detector_config: BallDetectorConfig) -> None:
        """Update ball detector settings dynamically without restarting video streams."""
        if hasattr(self.shared_ball_detector, "update_config"):
            self.shared_ball_detector.update_config(
                enabled=ball_detector_config.enabled,
                confidence_threshold=ball_detector_config.confidence_threshold,
                model_name=ball_detector_config.model_name,
                classes=ball_detector_config.classes,
                aspect_ratio=ball_detector_config.aspect_ratio,
            )
        else:
            self.shared_ball_detector = BallDetectorYOLO(
                model_name=ball_detector_config.model_name,
                confidence_threshold=ball_detector_config.confidence_threshold,
                classes=ball_detector_config.classes,
                enabled=ball_detector_config.enabled,
                aspect_ratio=ball_detector_config.aspect_ratio,
            )
            for pipe in self.pipelines.values():
                pipe.ball_detector = self.shared_ball_detector

        # Invalidate transform caches across pipelines so target changes apply immediately
        for pipe in self.pipelines.values():
            pipe.invalidate_transform()

        logger.info(
            f"Ball detector dynamically updated: enabled={ball_detector_config.enabled}, "
            f"model='{ball_detector_config.model_name}', conf={ball_detector_config.confidence_threshold:.2f}, "
            f"aspect_ratio={ball_detector_config.aspect_ratio}"
        )

    def update_camera_weights(self, camera_configs: List[CameraConfig]) -> None:
        """Update camera weights dynamically without restarting video streams."""
        for cfg in camera_configs:
            if cfg.id in self.pipelines:
                self.pipelines[cfg.id].config.weights = cfg.weights
        logger.info("Camera weights dynamically updated across all pipelines.")

    def start_all(self) -> None:
        """Starts all background grabbers."""
        logger.info(f"Starting {len(self.pipelines)} camera ingestion pipelines...")
        for pipeline in self.pipelines.values():
            pipeline.start()

    def stop_all(self) -> None:
        """Stops all background grabbers."""
        logger.info("Stopping all camera pipelines...")
        for pipeline in self.pipelines.values():
            pipeline.stop()

    def process_tick(self) -> Dict[str, CVResult]:
        """Pulls latest frames and runs CV inference across all cameras.

        Returns:
            Dict[str, CVResult]: Mapping of camera_id to CVResult.
        """
        results: Dict[str, CVResult] = {}
        for cam_id, pipeline in self.pipelines.items():
            results[cam_id] = pipeline.process_latest()
        return results
