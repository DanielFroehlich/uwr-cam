# cam-sim: Camera Stream Simulator
## Component Design Specification

---

### 1. Overview & Purpose

`cam-sim` is a standalone simulation tool designed for the **uwr-cam** project. Its purpose is to simulate realistic multi-camera IP setups for **Underwater Rugby (UWR)** without requiring physical pool cameras or live network hardware.

`cam-sim` operates as an **RTSP Streaming Server** (listening on port `8554` by default). External clients—such as **OBS Studio**, the **`uwr-cam` detection engine**, or external media players—connect directly to the server to pull video feeds on demand.

Given a set of input MP4 video recordings, `cam-sim` simulates **2 to 6 IP cameras**. For each simulated camera, `cam-sim` decodes a single MP4 file and simultaneously serves **two RTSP endpoints** at different resolutions:
1. **Low-Resolution Feed** (e.g., `rtsp://<server-ip>:8554/cam1/low`): Pulled by `uwr-cam` for real-time computer vision inference (e.g., 640x360 @ 30 fps).
2. **High-Resolution Feed** (e.g., `rtsp://<server-ip>:8554/cam1/high`): Pulled by **OBS Studio** for high-definition broadcast output (e.g., 1080p or 4K @ 30 fps).

**Key Capabilities**:
- **Decoupled Client-Pull Architecture**: `cam-sim` does not push to client IP addresses; it listens as a server, exactly matching physical IP cameras.
- **Distributed Deployment Ready**: Can run on the same machine as OBS or on separate physical laptops across a local network (LAN) by binding to `0.0.0.0:8554`.
- **Dual-Feed Scaling**: Generates low-res and high-res streams from a single MP4 decode pass using FFmpeg filtergraphs.
- **Containerized for macOS Apple Silicon**: Packaged as a lightweight Podman container (`linux/arm64`).

---

### 2. Architecture & Data Flow

```
                      +-------------------------------+
                      |      cam-sim/config.yaml      |
                      +---------------+---------------+
                                      |
                                      v
                      +-------------------------------+
                      |       cam-sim Supervisor      |
                      |   - Launches MediaMTX Server  |
                      |   - Manages FFmpeg Publishers |
                      +---------------+---------------+
                                      |
        +-----------------------------+-----------------------------+
        |                                                           |
        v                                                           v
+-------------------------------+                           +-------------------------------+
|    Camera 1 Pipeline          |                           |    Camera N Pipeline (2..6)   |
|                               |                           |                               |
| Source: cam1_match.mp4        |                           | Source: camN_match.mp4        |
|               | (Single Decode)                           |               | (Single Decode)
|               v                                           |               v
| FFmpeg Filter Graph:          |                           | FFmpeg Filter Graph:          |
| split=2 -> scale [low & high] |                           | split=2 -> scale [low & high] |
|               |                                           |               |
|               +-----------------------+                   |               +-----------------------+
|               |                       |                   |               |                       |
|   RTSP Publish (TCP)      RTSP Publish (TCP)              |   RTSP Publish (TCP)      RTSP Publish (TCP)
|   localhost:8554/cam1/low localhost:8554/cam1/high        |   localhost:8554/camN/low localhost:8554/camN/high
+---------------+-----------------------+-------------------+---------------+-----------------------+
                |                       |                                   |                       |
                +-----------------------+---------+-------------------------+-----------------------+
                                                  |
                                                  v
                               +-------------------------------------+
                               |     Embedded RTSP Server (MediaMTX) |
                               |     Listening on 0.0.0.0:8554       |
                               +------------------+------------------+
                                                  |
                         +------------------------+------------------------+
                         | (RTSP Client Requests)                          | (RTSP Client Requests)
                         v                                                 v
           +---------------------------+                     +---------------------------+
           |       uwr-cam Engine      |                     |         OBS Studio        |
           |                           |                     |                           |
           | Connects to:              |                     | Connects to:              |
           | rtsp://<host>:8554/.../low|                     | rtsp://<host>:8554/.../high
           +---------------------------+                     +---------------------------+
```

---

### 3. Key Design Decisions

#### 3.1. RTSP Server via Embedded MediaMTX
- **Decision**: Integrate [MediaMTX](https://github.com/bluenviron/mediamtx) as the core RTSP streaming server within the container.
- **Rationale**: 
  - MediaMTX is a battle-tested, zero-dependency, ultra-low latency RTSP server compiled as a single static Go binary natively for `linux/arm64`.
  - It handles multi-client fan-out (multiple clients can read the same camera stream simultaneously), supports both TCP interleaved and UDP RTP transports, and requires zero complex configuration.
  - Sub-millisecond internal routing overhead ensures compliance with our strict latency constraints.

#### 3.2. Single-Decode Dual-Output Transcoding Pipeline
- **Decision**: Use FFmpeg with a single decode step and `filter_complex split=2` followed by independent scaling and encoding.
- **Rationale**: Decoding high-definition MP4 files twice per camera wastes CPU cycles. Transcoding both feeds in a single process per camera guarantees temporal synchronization between the low-res tracking stream and high-res broadcast stream while minimizing CPU load on Apple Silicon.

#### 3.3. Real-Time Endless Loop Streaming
- **Decision**: Pass `-re` (real-time input pacing) and `-stream_loop -1` (infinite looping) to FFmpeg.
- **Rationale**: Keeps PTS/DTS timestamps flowing continuously without resets or desyncs, mimicking real hardware IP cameras.

#### 3.4. Low-Latency Encoding Parameters
- **Video Codec**: `libx264` (H.264)
- **Preset**: `ultrafast`
- **Tune**: `zerolatency`
- **GOP Size**: 30 frames (1 keyframe per second at 30 fps) for instantaneous client connection without waiting for long keyframe intervals.
- **Transport**: RTSP over TCP (`-rtsp_transport tcp`) for reliable local inter-process publishing to MediaMTX.

---

### 4. Configuration Schema (`cam-sim/config.yaml`)

```yaml
# cam-sim Configuration
version: "1.0"

# Server network settings
server:
  rtsp_port: 8554                 # RTSP port exposed to clients
  bind_address: "0.0.0.0"         # Listen on all interfaces

# Default encoding parameters
defaults:
  loop: true                      # Loop input video infinitely
  video_codec: "libx264"
  preset: "ultrafast"
  tune: "zerolatency"
  gop_size: 30

# Simulated Cameras (2 to 6 cameras supported)
cameras:
  - id: "cam1"
    name: "Underwater Goal Left"
    source: "/videos/goal_left.mp4"
    feeds:
      low_res:
        enabled: true
        path: "/cam1/low"
        width: 640
        height: 360
        fps: 30
        bitrate: "1000k"
      high_res:
        enabled: true
        path: "/cam1/high"
        width: 1920
        height: 1080
        fps: 30
        bitrate: "6000k"

  - id: "cam2"
    name: "Underwater Goal Right"
    source: "/videos/goal_right.mp4"
    feeds:
      low_res:
        enabled: true
        path: "/cam2/low"
        width: 640
        height: 360
        fps: 30
        bitrate: "1000k"
      high_res:
        enabled: true
        path: "/cam2/high"
        width: 1920
        height: 1080
        fps: 30
        bitrate: "6000k"

  - id: "cam3"
    name: "Surface Center Overview"
    source: "/videos/center_overview.mp4"
    feeds:
      low_res:
        enabled: true
        path: "/cam3/low"
        width: 640
        height: 360
        fps: 30
        bitrate: "1000k"
      high_res:
        enabled: true
        path: "/cam3/high"
        width: 1920
        height: 1080
        fps: 30
        bitrate: "6000k"
```

---

### 5. FFmpeg Publishing Pipeline

For each configured camera, the `cam-sim` supervisor spawns an FFmpeg worker publishing directly to the embedded MediaMTX server:

```bash
ffmpeg -re -stream_loop -1 -i "${SOURCE_FILE}" \
  -filter_complex \
  "[0:v]split=2[v_high_in][v_low_in]; \
   [v_low_in]scale=w=${LOW_W}:h=${LOW_H}:force_original_aspect_ratio=decrease,pad=${LOW_W}:${LOW_H}:(ow-iw)/2:(oh-ih)/2,fps=${LOW_FPS}[v_low_out]; \
   [v_high_in]scale=w=${HIGH_W}:h=${HIGH_H}:force_original_aspect_ratio=decrease,pad=${HIGH_W}:${HIGH_H}:(ow-iw)/2:(oh-ih)/2,fps=${HIGH_FPS}[v_high_out]" \
  -map "[v_low_out]" \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -b:v ${LOW_BITRATE} \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:${RTSP_PORT}${LOW_PATH}" \
  -map "[v_high_out]" \
    -c:v libx264 -preset ultrafast -tune zerolatency -g 30 -b:v ${HIGH_BITRATE} \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:${RTSP_PORT}${HIGH_PATH}"
```

---

### 6. Component Directory Structure

```
uwr-cam/
├── REQUIREMENTS.md
└── cam-sim/
    ├── DESIGN.md                 # This specification document
    ├── Containerfile             # Podman ARM64 container definition
    ├── config.yaml               # Active simulator configuration
    ├── config.example.yaml       # Template configuration with comments
    ├── run.sh                    # Helper runner script
    ├── src/
    │   ├── __init__.py
    │   ├── main.py               # Supervisor: starts MediaMTX and camera workers
    │   ├── config.py             # YAML config loader and validator
    │   └── pipeline.py           # FFmpeg command builder and subprocess manager
    └── videos/                   # Volume mount directory for source MP4 files
        └── .gitkeep
```

---

### 7. Containerization & Deployment (Podman on Apple Silicon)

#### 7.1. Container Build
The container installs `mediamtx` (native `linux/arm64` binary), `ffmpeg`, and Python 3.12:

```dockerfile
FROM python:3.12-slim-bookworm

# Install ffmpeg, curl, procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl procps \
    && rm -rf /var/lib/apt/lists/*

# Install MediaMTX ARM64
ARG MEDIAMTX_VERSION=v1.9.3
RUN curl -L -s https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/mediamtx_${MEDIAMTX_VERSION}_linux_arm64.tar.gz \
    | tar -xz -C /usr/local/bin/ mediamtx

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8554
ENTRYPOINT ["python3", "-m", "src.main"]
```

#### 7.2. Podman Execution
To expose the RTSP server to OBS and `uwr-cam`:

```bash
# Build ARM64 container image
podman build -t cam-sim -f cam-sim/Containerfile cam-sim

# Run container exposing RTSP port 8554
podman run --rm -it \
  --name cam-sim \
  -p 8554:8554 \
  -v ./cam-sim/videos:/videos:ro \
  -v ./cam-sim/config.yaml:/app/config.yaml:ro \
  cam-sim
```

#### 7.3. Client Connection URLs
Once running, clients connect using standard RTSP URLs:
- **Local Machine**:
  - `rtsp://localhost:8554/cam1/low` (for `uwr-cam`)
  - `rtsp://localhost:8554/cam1/high` (for OBS Media/VLC source)
- **Separate Physical Laptop across LAN**:
  - `rtsp://192.168.1.50:8554/cam1/low`
  - `rtsp://192.168.1.50:8554/cam1/high`

---

### 8. Verification & Test Plan

1. **Configuration Validation**:
   - Verify config parsing, duplicate path detection (e.g. two cameras claiming `/cam1/low`), and parameter boundaries (2 to 6 cameras).
2. **Process Supervision**:
   - Verify MediaMTX starts cleanly and binds to `:8554`.
   - Verify FFmpeg workers launch and publish to their designated RTSP paths.
   - Verify clean shutdown (SIGINT/SIGTERM terminates all FFmpeg workers and MediaMTX).
3. **Synthetic End-to-End Test**:
   - Generate a short synthetic MP4 (`testsrc` with timestamp overlay).
   - Start `cam-sim`.
   - Test stream playback and frame resolution using `ffprobe` and OpenCV `cv2.VideoCapture`:
     ```bash
     ffprobe -v error -show_entries stream=width,height,r_frame_rate rtsp://localhost:8554/cam1/low
     ffprobe -v error -show_entries stream=width,height,r_frame_rate rtsp://localhost:8554/cam1/high
     ```
