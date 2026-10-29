"""Unit tests for FFmpeg pipeline command construction."""

import pytest

from src.config import (
    CameraConfig,
    CameraFeedsConfig,
    DefaultsConfig,
    FeedConfig,
    ServerConfig,
)
from src.pipeline import build_ffmpeg_camera_command, build_scale_filter


@pytest.fixture
def sample_camera() -> CameraConfig:
    return CameraConfig(
        id="cam1",
        name="Pool Cam 1",
        source="/videos/match.mp4",
        feeds=CameraFeedsConfig(
            low_res=FeedConfig(
                enabled=True,
                path="/cam1/low",
                width=640,
                height=360,
                fps=30,
                bitrate="1000k",
            ),
            high_res=FeedConfig(
                enabled=True,
                path="/cam1/high",
                width=1920,
                height=1080,
                fps=30,
                bitrate="6000k",
            ),
        ),
    )


def test_build_scale_filter():
    """Verify that scale and pad filter preserve aspect ratio and enforce fps."""
    filt = build_scale_filter(640, 360, 30)
    assert "scale=w=640:h=360:force_original_aspect_ratio=decrease" in filt
    assert "pad=640:360:(ow-iw)/2:(oh-ih)/2" in filt
    assert "fps=30" in filt


def test_dual_feed_command_structure(sample_camera):
    """Verify that dual-feed camera generates correct split filter and 2 RTSP outputs."""
    server = ServerConfig(rtsp_port=8554, bind_address="0.0.0.0")
    defaults = DefaultsConfig(
        loop=True,
        video_codec="libx264",
        preset="ultrafast",
        tune="zerolatency",
        gop_size=30,
    )

    cmd = build_ffmpeg_camera_command(sample_camera, server, defaults)

    # Must include input loop and real-time flags
    assert "-stream_loop" in cmd
    assert "-re" in cmd
    assert "-i" in cmd
    assert sample_camera.source in cmd

    # Must contain filter_complex with split=2
    assert "-filter_complex" in cmd
    fc_idx = cmd.index("-filter_complex")
    fc_val = cmd[fc_idx + 1]
    assert "[0:v]split=2" in fc_val
    assert "[v_low_out]" in fc_val
    assert "[v_high_out]" in fc_val

    # Must contain two RTSP destination URLs with TCP transport
    rtsp_low = "rtsp://127.0.0.1:8554/cam1/low"
    rtsp_high = "rtsp://127.0.0.1:8554/cam1/high"
    assert rtsp_low in cmd
    assert rtsp_high in cmd

    # Verify codecs, zerolatency tuning, and repeat-headers for fast lock-on
    assert "libx264" in cmd
    assert "zerolatency" in cmd
    assert "ultrafast" in cmd
    assert "yuv420p" in cmd
    assert "-x264-params" in cmd
    param_idx = cmd.index("-x264-params")
    assert "repeat-headers=1" in cmd[param_idx + 1]
    assert "scenecut=0" in cmd[param_idx + 1]


def test_single_feed_command(sample_camera):
    """Verify command structure when only one feed is enabled."""
    sample_camera.feeds.high_res.enabled = False
    server = ServerConfig(rtsp_port=8554)
    defaults = DefaultsConfig()

    cmd = build_ffmpeg_camera_command(sample_camera, server, defaults)

    # Should use simple -vf filter instead of split=2 filter_complex
    assert "-vf" in cmd
    assert "-filter_complex" not in cmd
    assert "rtsp://127.0.0.1:8554/cam1/low" in cmd
    assert "rtsp://127.0.0.1:8554/cam1/high" not in cmd
