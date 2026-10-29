# Underwater Rugby Automated Camera Switcher (uwr-cam)
## Requirements & System Specification Document

---

### 1. Executive Summary & Objective

The objective of the **uwr-cam** project is to build an automated, real-time video stream analyzer and production switcher controller specifically tailored for **Underwater Rugby (UWR)**.

The system connects to multiple IP cameras acting as **RTSP streaming servers** (streaming H.264 video via RTP). The `uwr-cam` engine concurrently pulls low-resolution video feeds, continuously analyzes each feed using computer vision to detect the UWR ball and player motion density, and dynamically commands **OBS Studio** (via WebSocket) to switch the live broadcast scene to the camera showing the ball or the most active play. High-resolution feeds (1K/4K) are pulled directly by OBS Studio for final broadcast output.

The project also includes a dedicated **Camera Simulator Tool (`cam-sim`)** that acts as an RTSP server, reading MP4 video files and exposing dual-resolution streams in an endless loop, enabling end-to-end testing, model training, and tuning without requiring physical underwater cameras.

---

### 2. Component Model & System Architecture

```
+-----------------------------------------------------------------------------------+
|                            DEVELOPMENT & TESTING TOOLING                          |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                       cam-sim (Camera Simulator Tool)                     |   |
|   |  - Dedicated component directory (`cam-sim/`) with independent config     |   |
|   |  - RTSP Streaming Server (listening on e.g. port 8554)                    |   |
|   |  - Ingests recorded MP4 test files (UWR match footage) in endless loop    |   |
|   |  - Dual-Feed Output per Camera via up/downsampling:                       |   |
|   |    * Low-Res:  rtsp://<host>:8554/<cam-id>/low                            |   |
|   |    * High-Res: rtsp://<host>:8554/<cam-id>/high                           |   |
|   |  - Multi-camera simulation: Simulates 2 to 6 cameras concurrently         |   |
|   +-------------------------------------+-------------------------------------+   |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          | (RTSP / RTP Streams pulled by clients)
                                          v
+-----------------------------------------------------------------------------------+
|                                 PRODUCTION ENVIRONMENT                            |
|                                                                                   |
|                  +----------------------------------------------+                 |
|                  |     Physical IP Cameras (2 to 6 Cameras)     |                 |
|                  |     RTSP Servers (Low-Res & High-Res Feeds)  |                 |
|                  +----------------------+-----------------------+                 |
+-----------------------------------------|-----------------------------------------+
                                          |
                +-------------------------+-------------------------+
                | rtsp://.../<cam>/low                              | rtsp://.../<cam>/high
                | (Client pull)                                     | (Client pull)
                v                                                   v
 +-----------------------------+                     +-----------------------------+
 |      uwr-cam Engine         |                     |         OBS Studio          |
 |                             |                     |                             |
 |  +-----------------------+  |                     |  +-----------------------+  |
 |  | RTSP Stream Ingestion |  |                     |  | High-Res RTSP Sources |  |
 |  | Frame Grabber Manager |  |                     |  | (1 Scene per Camera)  |  |
 |  +-----------+-----------+  |                     |  +-----------^-----------+  |
 |              |              |                     |              |              |
 |              v              |                     |              |              |
 |  +-----------------------+  |                     |              |              |
 |  | CV Detection:         |  |                     |              |              |
 |  | - UWR Ball Detection  |  |                     |              |              |
 |  | - Motion Density      |  |                     |              |              |
 |  +-----------+-----------+  |                     |              |              |
 |              |              |                     |              |              |
 |              v              |                     |              |              |
 |  +-----------------------+  |                     |              |              |
 |  | Arbitration Logic     |  |                     |              |              |
 |  | (Hysteresis & Dwell)  |  |                     |              |              |
 |  +-----------+-----------+  |                     |              |              |
 |              |              |                     |              |              |
 |              v              |                     |              |              |
 |  +-----------------------+  |   obs-websocket     |              |              |
 |  | OBS WebSocket Client  |==|====================>| SetCurrent   |              |
 |  +-----------------------+  |   (Scene Switch)    | ProgramScene |              |
 +--------------^--------------+                     +-----------------------------+
                |
 +--------------+--------------+
 |  Dynamic YAML Configuration |
 |  (Hot-Reloading at Runtime) |
 +-----------------------------+
```

---

### 3. Functional Requirements

#### 3.1. Video Feed Ingestion
- **FR-1.1 Camera Count**: Concurrently ingest from **2 to 6** IP cameras.
- **FR-1.2 Client-Pull Dual-Stream Architecture**:
  - Each camera (physical or simulated) acts as an RTSP server exposing two endpoints:
    1. **Low-Resolution Feed** (e.g. `rtsp://<camera-ip>:8554/<cam-id>/low`): Pulled by `uwr-cam` for real-time computer vision inference.
    2. **High-Resolution Feed** (e.g. `rtsp://<camera-ip>:8554/<cam-id>/high`): Pulled directly by OBS Studio for high-definition broadcast output.
- **FR-1.3 Protocol Support**: Standard RTSP (Real Time Streaming Protocol) carrying RTP/H.264 (supporting both TCP interleaved and UDP RTP transport).
- **FR-1.4 Low Latency Ingestion**: Drop buffered/queued frames immediately if processing falls behind to guarantee zero latency accumulation (real-time stream analysis).
- **FR-1.5 Fault Tolerance & Auto-Reconnection**: Automatically attempt reconnection if an RTSP connection drops or experiences network interruptions, without crashing the application.

#### 3.2. Detection & Analysis Engine (Underwater Rugby)
- **FR-2.1 Ball Detection (Primary Metric)**:
  - Detect presence and tracking confidence of the Underwater Rugby ball in each active camera frame.
  - Model / feature extractor to be trained and evaluated using domain-specific training material provided in MP4 format.
- **FR-2.2 Motion Density Analysis (Secondary / Fallback Metric)**:
  - Measure localized movement and activity density (e.g., optical flow, frame differencing, or motion heatmaps) across cameras.
  - In situations where the ball is occluded, under scrums, or lost from view, rank cameras based on motion intensity.
- **FR-2.3 Composite Score Calculation**:
  - Combine ball detection confidence ($C_{\text{ball}}$) and motion intensity ($M$) using configurable weights:
    $$\text{Score}_i = w_{\text{ball}} \cdot C_{\text{ball}, i} + w_{\text{motion}} \cdot M_i$$

#### 3.3. Decision & Switching Logic
- **FR-3.1 Hysteresis & Minimum Dwell Time**:
  - Enforce a minimum display time on any given scene (e.g., 2–3 seconds) to prevent rapid or jarring camera cuts ("camera flapping").
  - Require candidate scenes to exceed the current scene's score by a configurable threshold ($\Delta \text{score}$) before triggering a switch.
- **FR-3.2 Fallback Default Scene**:
  - Revert to a designated fallback/overview scene if no camera registers confidence or motion above baseline thresholds.
- **FR-3.3 Manual Override Support**:
  - Allow pausing automated switching to enable manual director control when desired.

#### 3.4. OBS Studio Integration
- **FR-4.1 obs-websocket Protocol (v5.x)**:
  - Connect to OBS Studio using the standard `obs-websocket` v5 protocol with password authentication.
- **FR-4.2 Scene-Level Switching**:
  - Switch broadcast display via `SetCurrentProgramScene` to the target scene mapped to the winning camera.
- **FR-4.3 Connection Monitoring**:
  - Monitor connection health, auto-reconnect to OBS if disconnected, and verify scene existence upon configuration changes.

#### 3.5. Configuration & Dynamic Reconfiguration
- **FR-5.1 Centralized YAML Configuration**:
  - All system settings must reside in a single, well-documented YAML configuration file (`config.yaml`).
  - Configuration includes:
    - Camera endpoints (RTSP URLs for low-res streams).
    - Camera-to-OBS scene mappings (e.g., Camera 1 -> "Scene - Cam 1").
    - OBS WebSocket connection parameters (host, port, password).
    - Computer vision thresholds, weights, dwell times, and hysteresis parameters.
- **FR-5.2 Dynamic Runtime Reloading**:
  - Monitor `config.yaml` for filesystem modifications.
  - Apply updated configuration parameters dynamically at runtime without terminating the main application process.
  - **Graceful Reconfiguration Window**: A brief stream ingestion / processing interruption of **up to 10 seconds** is acceptable while re-initializing camera ingestion pipelines or applying updated stream URLs.

#### 3.6. Tooling: cam-sim (Camera Stream Simulator)
- **FR-6.1 Component Isolation & Independent Configuration**:
  - Located in its own dedicated component directory (`cam-sim/`).
  - Governed by its own dedicated YAML configuration file (`cam-sim/config.yaml`).
- **FR-6.2 RTSP Server Architecture**:
  - Hosts an embedded RTSP server exposing a standard RTSP port (default `8554`).
  - Allows external clients (OBS Studio, `uwr-cam`, VLC, ffplay) to connect and pull streams on demand.
- **FR-6.3 MP4 Ingestion & Endless Loop**:
  - Accept an MP4 video file as the source for each simulated camera.
  - Stream continuously in an endless loop at the video's native framerate and timestamp cadence.
- **FR-6.4 Dual-Feed Generation per Camera (Up/Downsampling)**:
  - From a single source MP4 file, generate **two simultaneous RTSP stream paths** per camera:
    1. **Low-Resolution Feed** (e.g. `/cam1/low`): Scaled (downsampled or matched) for the `uwr-cam` detection engine.
    2. **High-Resolution Feed** (e.g. `/cam1/high`): Scaled (original, 1K, or 4K up/downsampled) for OBS Studio broadcast ingestion.
  - Provide independent path, resolution, framerate, and bitrate configuration for each feed.
- **FR-6.5 Multi-Camera Simulation**:
  - Support simulating 2 to 6 cameras concurrently (yielding 4 to 12 simultaneous RTSP streams) hosted on a single server instance.
- **FR-6.6 Containerized Execution**:
  - Package `cam-sim` as a standalone Podman container running on macOS Apple Silicon (`arm64`), exposing port `8554`.

---

### 4. Non-Functional Requirements

- **NFR-1 Latency**:
  - Maximum end-to-end latency of **100 ms** from frame capture to switching command dispatch.
- **NFR-2 Implementation Stack**:
  - **Phase 1 (Rapid Prototyping & Iteration)**: Implemented in **Python** using asynchronous I/O (`asyncio`), low-latency video decoding (OpenCV / PyAV / FFmpeg), and WebSocket client libraries.
  - **Architecture Portability**: Design clean, decoupled abstractions (ingestion, detection, arbitration, OBS client) to allow future porting to **Go** for ultra-low latency execution if required.
- **NFR-3 Resource Efficiency**:
  - Low-resolution processing must remain lightweight to sustain multi-stream processing (up to 6 streams simultaneously) on standard commodity hardware.
- **NFR-4 Observability**:
  - Structured console logging with configurable log levels (DEBUG, INFO, WARNING, ERROR).
  - Optional visual preview/debug window displaying bounding boxes, motion metrics, and real-time score ranking.
- **NFR-5 Deployment & Containerization (Podman on Apple Silicon)**:
  - **Container Runtime**: All components (both the `uwr-cam` engine and the `cam-sim` tool) must be containerized using **Podman** (via `Containerfile` / `Dockerfile`).
  - **Target Hardware Architecture**: Built, tested, and optimized natively for **macOS on Apple Silicon (`linux/arm64` / `aarch64`)**.
  - **Container Networking & Port Exposure**:
    - `cam-sim`: Publishes RTSP server port `8554` (e.g. `-p 8554:8554`).
    - `uwr-cam`: Connects outbound via TCP to RTSP camera servers and OBS Studio WebSocket (`host.containers.internal` / host IP).
  - **Orchestration & Workflow**: Provide `podman-compose` or Podman Pod deployment manifests for frictionless one-command launch of the system and mock streaming pipelines.

---

### 5. Next Steps & Implementation Roadmap

1. **Phase 1: cam-sim Implementation**:
   - Create ARM64 `Containerfile` for `cam-sim` with embedded RTSP server and FFmpeg.
   - Implement YAML parser, FFmpeg publishing pipeline, and process supervisor.
   - Validate with synthetic test video and client connection tests (OpenCV / ffprobe / VLC).
2. **Phase 2: uwr-cam Configuration & OBS WebSocket Client**:
   - Define `config.yaml` schema and hot-reload watcher.
   - Implement `obs-websocket` v5 client for scene switching and connection management.
3. **Phase 3: Video Ingestion & Motion Detection**:
   - Implement low-latency RTSP frame grabber with frame dropping.
   - Implement motion density calculation (optical flow / frame differencing).
4. **Phase 4: Ball Detection & Scoring Engine**:
   - Ingest training MP4 footage from user.
   - Evaluate and integrate ball detector (e.g., color/contour segmentation or lightweight YOLO model).
   - Implement score aggregation and switching arbitration (dwell time / hysteresis).
