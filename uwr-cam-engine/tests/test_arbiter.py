"""Unit tests for SwitchingArbiter and composite scoring logic."""

from __future__ import annotations

import pytest

from src.arbiter import SwitchingArbiter
from src.config import CameraConfig, CameraWeights, SwitchingConfig
from src.pipeline_manager import CVResult


@pytest.fixture
def sample_cameras():
    return [
        CameraConfig(
            id="cam1",
            name="Goal Left",
            rtsp_url="rtsp://localhost:8554/cam1/low",
            obs_scene="cam1",
            weights=CameraWeights(ball=0.7, motion=0.3),
        ),
        CameraConfig(
            id="cam2",
            name="Goal Right",
            rtsp_url="rtsp://localhost:8554/cam2/low",
            obs_scene="cam2",
            weights=CameraWeights(ball=0.7, motion=0.3),
        ),
    ]


@pytest.fixture
def sample_switching_config():
    return SwitchingConfig(
        fallback_scene="base",
        min_dwell_time_sec=2.0,
        hysteresis_threshold=0.15,
        fallback_timeout_sec=5.0,
        manual_override=False,
    )


def test_arbiter_initial_state(sample_switching_config, sample_cameras):
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras)
    assert arbiter.current_scene == "base"
    assert arbiter.fallback_scene == "base"


def test_arbiter_switches_from_base_to_active_cam(sample_switching_config, sample_cameras):
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras, initial_scene="base")
    t0 = 100.0
    arbiter.last_switch_time = t0 - 5.0  # Dwell expired

    # cam1 has high score (0.6)
    results = {
        "cam1": CVResult(camera_id="cam1", score=0.6),
        "cam2": CVResult(camera_id="cam2", score=0.1),
    }

    target = arbiter.evaluate(results, now=t0)
    assert target == "cam1"
    assert arbiter.current_scene == "cam1"
    assert arbiter.last_switch_time == t0


def test_arbiter_enforces_dwell_time(sample_switching_config, sample_cameras):
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras, initial_scene="cam1")
    t0 = 100.0
    arbiter.last_switch_time = t0  # Just switched

    # cam2 suddenly has higher score (0.9 vs 0.2)
    results = {
        "cam1": CVResult(camera_id="cam1", score=0.2),
        "cam2": CVResult(camera_id="cam2", score=0.9),
    }

    # Only 1.0s elapsed (dwell is 2.0s) -> should NOT switch
    target = arbiter.evaluate(results, now=t0 + 1.0)
    assert target is None
    assert arbiter.current_scene == "cam1"

    # 2.5s elapsed (dwell expired) -> should switch
    target_after_dwell = arbiter.evaluate(results, now=t0 + 2.5)
    assert target_after_dwell == "cam2"
    assert arbiter.current_scene == "cam2"


def test_arbiter_enforces_hysteresis_threshold(sample_switching_config, sample_cameras):
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras, initial_scene="cam1")
    t0 = 100.0
    arbiter.last_switch_time = t0 - 5.0  # Dwell expired

    # cam1 score is 0.50. Hysteresis margin is 0.15. Required to switch: > 0.65.
    # cam2 has 0.60 (only +0.10 higher) -> should NOT switch
    results = {
        "cam1": CVResult(camera_id="cam1", score=0.50),
        "cam2": CVResult(camera_id="cam2", score=0.60),
    }
    target = arbiter.evaluate(results, now=t0)
    assert target is None
    assert arbiter.current_scene == "cam1"

    # cam2 increases to 0.70 (+0.20 higher) -> exceeds hysteresis -> SWITCH
    results["cam2"] = CVResult(camera_id="cam2", score=0.70)
    target = arbiter.evaluate(results, now=t0)
    assert target == "cam2"
    assert arbiter.current_scene == "cam2"


def test_arbiter_fallback_timeout_on_inactivity(sample_switching_config, sample_cameras):
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras, initial_scene="cam1")
    t0 = 100.0
    arbiter.last_switch_time = t0 - 10.0
    arbiter.last_activity_time = t0 - 6.0  # Inactive for 6.0s (timeout is 5.0s)

    # All cameras below inactivity threshold (0.05)
    results = {
        "cam1": CVResult(camera_id="cam1", score=0.01),
        "cam2": CVResult(camera_id="cam2", score=0.02),
    }

    target = arbiter.evaluate(results, now=t0)
    assert target == "base"
    assert arbiter.current_scene == "base"


def test_arbiter_manual_override_suppresses_switching(sample_switching_config, sample_cameras):
    sample_switching_config.manual_override = True
    arbiter = SwitchingArbiter(sample_switching_config, sample_cameras, initial_scene="base")
    t0 = 100.0
    arbiter.last_switch_time = t0 - 10.0

    # Massive activity on cam1
    results = {
        "cam1": CVResult(camera_id="cam1", score=0.99),
        "cam2": CVResult(camera_id="cam2", score=0.0),
    }

    target = arbiter.evaluate(results, now=t0)
    assert target is None
    assert arbiter.current_scene == "base"
