"""Switching arbitration state machine with hysteresis, dwell time, and fallback protection."""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from .config import CameraConfig, SwitchingConfig
from .pipeline_manager import CVResult

logger = logging.getLogger(__name__)


class SwitchingArbiter:
    """Arbitrates camera switching decisions based on composite scores.

    Implements:
    - FR-3.1: Minimum dwell time enforcement (anti-flapping)
    - FR-3.1: Hysteresis margin requiring candidate camera to significantly beat current camera
    - FR-3.2: Revert to fallback default scene on sustained inactivity
    - FR-3.3: Manual override suppression
    """

    def __init__(
        self,
        config: SwitchingConfig,
        camera_configs: list[CameraConfig],
        initial_scene: Optional[str] = None,
        inactivity_threshold: float = 0.05,
    ) -> None:
        self.config = config
        self.camera_map: Dict[str, str] = {c.id: c.obs_scene for c in camera_configs}
        self.scene_to_cam: Dict[str, str] = {c.obs_scene: c.id for c in camera_configs}
        self.fallback_scene = config.fallback_scene
        self.inactivity_threshold = inactivity_threshold

        self.current_scene: str = initial_scene or self.fallback_scene
        self.last_switch_time: float = time.time()
        self.last_activity_time: float = time.time()

    def update_current_scene(self, scene_name: str) -> None:
        """Synchronizes arbiter state with actual active program scene from OBS."""
        if self.current_scene != scene_name:
            logger.debug(f"Arbiter scene state updated: '{self.current_scene}' -> '{scene_name}'")
            self.current_scene = scene_name
            self.last_switch_time = time.time()

    def evaluate(self, results: Dict[str, CVResult], now: Optional[float] = None) -> Optional[str]:
        """Evaluates current camera scores and returns target scene if a switch is required.

        Args:
            results: Mapping of camera_id to CVResult.
            now: Current epoch timestamp (optional, defaults to time.time()).

        Returns:
            Optional[str]: Target scene name if a switch should occur, None otherwise.
        """
        current_time = now if now is not None else time.time()

        # Rule 1: Manual override active -> do not switch
        if self.config.manual_override:
            logger.debug("Manual override active. Skipping automated switching.")
            return None

        # Rule 2: Check for any activity across all cameras
        max_score = 0.0
        best_cam_id: Optional[str] = None
        for cam_id, res in results.items():
            if res.score > max_score:
                max_score = res.score
                best_cam_id = cam_id

        if max_score >= self.inactivity_threshold:
            self.last_activity_time = current_time

        # Rule 3: Dwell time check - anti-flapping guard
        time_since_last_switch = current_time - self.last_switch_time
        if time_since_last_switch < self.config.min_dwell_time_sec:
            logger.debug(
                f"Dwell time active ({time_since_last_switch:.2f}s < {self.config.min_dwell_time_sec:.2f}s). Holding scene."
            )
            return None

        # Rule 4: Fallback timeout check
        # If no camera has registered significant activity for fallback_timeout_sec,
        # revert to the fallback scene.
        time_since_activity = current_time - self.last_activity_time
        if time_since_activity >= self.config.fallback_timeout_sec:
            if self.current_scene != self.fallback_scene:
                logger.info(
                    f"No activity detected for {time_since_activity:.1f}s (timeout: {self.config.fallback_timeout_sec}s). "
                    f"Reverting to fallback scene: '{self.fallback_scene}'"
                )
                self.current_scene = self.fallback_scene
                self.last_switch_time = current_time
                return self.fallback_scene
            return None

        # Rule 5: Hysteresis evaluation
        if best_cam_id is None:
            return None

        target_scene = self.camera_map.get(best_cam_id)
        if not target_scene:
            return None

        # If already on the best scene, maintain
        if self.current_scene == target_scene:
            return None

        # Determine current camera score
        current_cam_id = self.scene_to_cam.get(self.current_scene)
        current_score = 0.0
        if current_cam_id and current_cam_id in results:
            current_score = results[current_cam_id].score

        # Candidate must beat current camera by at least hysteresis_threshold
        required_score = current_score + self.config.hysteresis_threshold
        if max_score > required_score:
            logger.info(
                f"Switching triggered: Camera '{best_cam_id}' (scene: '{target_scene}') "
                f"score {max_score:.2f} exceeds required {required_score:.2f} "
                f"(current '{self.current_scene}' score: {current_score:.2f}, margin: {self.config.hysteresis_threshold})"
            )
            self.current_scene = target_scene
            self.last_switch_time = current_time
            return target_scene

        return None
