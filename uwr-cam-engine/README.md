# uwr-cam-engine: Automated Camera Switcher Engine

[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black?logo=apple)](https://apple.com/)
[![OBS WebSocket: v5.x](https://img.shields.io/badge/OBS%20WebSocket-v5.x-teal)](https://github.com/obsproject/obs-websocket)

---

## 🏊 Overview

`uwr-cam-engine` is the central analysis and switching engine for the **Underwater Rugby Automated Camera Switcher (`uwr-cam`)**.

It connects to **OBS Studio** via WebSocket v5 RPC to dynamically switch broadcast scenes based on real-time computer vision analysis.

### Phase 3 Status: Completed ✅
- **Low-Latency Video Ingestion (`FrameGrabber`)**: Dedicated background thread per camera calling `cap.grab()` in an I/O-paced loop. Discards old buffered frames so the CV pipeline always operates on the zero-latency latest frame (FR-1.4). Includes automatic RTSP stream reconnection.
- **Adaptive Motion Density Detection (`MotionDetectorMOG2`)**: Uses OpenCV's MOG2 background subtractor to learn continuous water ripples as background and extract foreground masks corresponding to active player scrums (FR-2.2). Modularly decoupled via `MotionDetectorProtocol`.
- **Multi-Camera Orchestration (`PipelineManager`)**: Manages 2 to 6 camera pipelines concurrently, outputting a typed `CVResult` data contract prepared with ball detection placeholders for Phase 4.
- **Multi-Level Observability**: Granular per-frame debug tracing (`DEBUG`) and clean periodic status summaries (`INFO`) emitted every `status_interval_sec` seconds.
- **Dynamic YAML Configuration (`config.yaml`) & Hot-Reloading (`ConfigWatcher`)**: Pydantic schema validation and debounced file watcher that updates parameters without restarting the broadcast.
- **Decoupled OBS WebSocket v5 Client (`ObswsPythonClient`)**: Non-blocking thread delegation via `asyncio.to_thread()`, auto-reconnect, and scene verification.
- **Native Apple Silicon Architecture**: Native `.venv` execution unlocking MPS GPU acceleration and VideoToolbox hardware decoding.

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
