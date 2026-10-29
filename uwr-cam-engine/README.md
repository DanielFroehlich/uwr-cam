# uwr-cam-engine: Automated Camera Switcher Engine

[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black?logo=apple)](https://apple.com/)
[![OBS WebSocket: v5.x](https://img.shields.io/badge/OBS%20WebSocket-v5.x-teal)](https://github.com/obsproject/obs-websocket)

---

## 🏊 Overview

`uwr-cam-engine` is the central analysis and switching engine for the **Underwater Rugby Automated Camera Switcher (`uwr-cam`)**.

It connects to **OBS Studio** via WebSocket v5 RPC to dynamically switch broadcast scenes based on real-time computer vision analysis.

### System Status: All Core Phases Complete ✅
- **Phase 1 (`cam-sim`)**: Multi-camera RTSP simulator streaming dual-resolution feeds in an endless loop via embedded MediaMTX and FFmpeg.
- **Phase 2 (`uwr-cam-engine` Config & OBS)**: Dynamic YAML configuration with Pydantic validation, debounced hot-reloading (`ConfigWatcher`), and decoupled OBS WebSocket v5 controller (`ObswsPythonClient`).
- **Phase 3 (Video Ingestion & Motion Detection)**: Dedicated threaded `FrameGrabber` with zero-latency frame dropping and `MotionDetectorMOG2` adaptive background subtraction filtering water ripples.
- **Phase 4 (Ball Detection & Switching Arbitration)**: Dataset extraction utility (`scripts/extract_frames.py`), `BallDetectorYOLO` deep learning stub, composite camera scoring ($S = w_{\text{ball}} \cdot B + w_{\text{motion}} \cdot M$), and `SwitchingArbiter` state machine with dwell time, hysteresis, fallback, and manual override.
- **Automated Test Suite**: 43 unit tests across all modules (100% passing in ~0.65s).

---

## 🚀 Quickstart

### 1. Check Connectivity
Verify that OBS Studio (port 4455) and `cam-sim` (port 8554) are running:
```bash
./uwr-cam-engine/run.sh status
```

### 2. Run the Engine Natively (Recommended for Apple Silicon)
```bash
./uwr-cam-engine/run.sh start
```

### 3. Run Automated Tests
```bash
./uwr-cam-engine/run.sh test
```

---

## ⚙️ Configuration Reference (`config.yaml`)

```yaml
version: "1.0"

# OBS Studio WebSocket v5 connection settings
obs:
  host: "localhost"
  port: 4455
  password: ""                     # Set password if OBS authentication is enabled
  timeout_sec: 3.0
  reconnect_interval_sec: 3.0

# Switching and arbitration parameters
switching:
  fallback_scene: "base"           # Default overview scene if tracking is lost
  min_dwell_time_sec: 2.5          # Minimum seconds to stay on a scene (anti-flapping)
  hysteresis_threshold: 0.15       # Delta score required to switch away from current camera
  fallback_timeout_sec: 5.0        # Seconds of inactivity before reverting to fallback
  manual_override: false           # Pause automated switching for director control

# Console logging level
logging:
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR

# Camera inputs (2 to 6 cameras)
cameras:
  - id: "cam1"
    name: "Underwater Goal Left"
    rtsp_url: "rtsp://localhost:8554/cam1/low"
    obs_scene: "cam1"
    weights:
      ball: 0.7
      motion: 0.3

  - id: "cam2"
    name: "Underwater Goal Right"
    rtsp_url: "rtsp://localhost:8554/cam2/low"
    obs_scene: "cam2"
    weights:
      ball: 0.7
      motion: 0.3

  - id: "cam3"
    name: "Surface Center Overview"
    rtsp_url: "rtsp://localhost:8554/cam3/low"
    obs_scene: "cam3"
    weights:
      ball: 0.7
      motion: 0.3
```

---

## 🧪 Testing & Validation

The component test suite includes 30 unit tests validating configuration parsing, boundary constraints, hot-reloading debouncing, error tolerance, and OBS client RPC switching:

```bash
PYTHONPATH=uwr-cam-engine ./uwr-cam-engine/.venv/bin/pytest -v uwr-cam-engine/tests
```
