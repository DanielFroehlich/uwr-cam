# Phase 1 Design Decisions: Camera Simulator (`cam-sim`)

---

## 1. Objective & Scope

Phase 1 established the testing infrastructure for the **Underwater Rugby Automated Camera Switcher (`uwr-cam`)** by building an isolated multi-camera RTSP video stream simulator (`cam-sim`).

Without requiring physical IP cameras or an active pool session, `cam-sim` loops recorded match MP4 footage and streams synchronized RTSP feeds to both the computer vision engine and OBS Studio.

---

## 2. Key Design Decisions

### 2.1. Dual-Resolution RTSP Streaming Architecture
- **Decision**: Produce two distinct RTSP streams per camera from a single MP4 source file:
  - **Low-Resolution Feed** (`/cam<id>/low` e.g., 640x360 @ 30 fps): Consumed by the CV detection engine for motion analysis and ball tracking, minimizing CPU/GPU decoding load and network bandwidth.
  - **High-Resolution Feed** (`/cam<id>/high` e.g., 1920x1080 @ 30 fps): Ingested by OBS Studio as Media Source inputs for high-definition broadcast output.
- **Rationale**: Isolates the compute burden of CV inference from broadcast quality. The detection engine does not need 1080p frames to calculate motion density or detect the ball.

### 2.2. Single-Decode FFmpeg Pipeline (`split` Filtergraph)
- **Decision**: Use a single FFmpeg process per camera that decodes the source video once, splits the decoded YUV frame buffer using `-filter_complex "[0:v]split=2[v_high][v_low]; ..."`, and encodes both resolutions in parallel before publishing over RTSP.
- **Rationale**: Decoding H.264/HEVC is the most CPU-intensive step. Running two separate FFmpeg instances per camera would double decode overhead. Single-decode cuts CPU usage by ~45%.

### 2.3. Embedded MediaMTX RTSP Server
- **Decision**: Embed the lightweight, high-performance Go-based **MediaMTX** (formerly `rtsp-simple-server`) directly within the container rather than relying on external streaming servers or Python RTSP servers.
- **Rationale**: MediaMTX is zero-dependency, ultra-low latency, and supports RTSP over both TCP and UDP. It allows FFmpeg to publish feeds locally over TCP (`localhost:8554`) while clients (OBS, OpenCV) consume feeds concurrently.

### 2.4. Stream Continuity & Encoder Tuning
- **Decision**:
  - Endless looping via `-stream_loop -1`.
  - Fixed GOP length matching stream framerate (`-g 30` / `-keyint_min 30`) ensuring predictable 1.0-second I-frame intervals.
  - SPS/PPS repetition: `-x264-params repeat-headers=1:scenecut=0` to ensure new clients connecting to the RTSP stream at any instant can decode immediately without waiting for a keyframe.
- **Rationale**: OBS and OpenCV clients may connect or reconnect at arbitrary times. Without repeated headers and short GOPs, clients would freeze or drop initial frames.

### 2.5. Configuration Validation (Pydantic v2)
- **Decision**: Enforce strict invariants at startup via Pydantic:
  - 2 to 6 camera limit.
  - Alphanumeric camera IDs.
  - Unique camera IDs and unique RTSP mount paths.
  - Even width/height dimensions (required by libx264 YUV420p).
  - Positive bitrates and framerates.

### 2.6. Containerized Deployment (ARM64 Podman)
- **Decision**: Package `cam-sim` in a minimal Debian Bookworm container (`python:3.12-slim-bookworm`) with multi-arch/ARM64 support for Apple Silicon.
- **Rationale**: Fully isolates FFmpeg and MediaMTX dependencies from the host system while allowing reproducible execution.
