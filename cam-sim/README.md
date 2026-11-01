# cam-sim: Underwater Rugby Multi-Camera RTSP Stream Simulator

`cam-sim` is a standalone tool designed to simulate multi-camera Underwater Rugby (UWR) setups for testing and development without requiring physical IP cameras.

It runs an embedded **MediaMTX RTSP streaming server** and uses **FFmpeg** to transcode single MP4 video recordings into **two simultaneous RTSP video feeds** per camera:
1. **Low-Resolution Feed** (e.g. `rtsp://<host>:8554/cam1/low`): 640x360 @ 30fps for real-time computer vision inference.
2. **High-Resolution Feed** (e.g. `rtsp://<host>:8554/cam1/high`): 1080p/4K @ 30fps for OBS Studio broadcast ingestion.

---

## 1. Prerequisites

- **macOS with Apple Silicon (ARM64)** or **Linux (aarch64 / x86_64)**
- **Podman** installed and running (`podman machine start` on macOS)

---

## 2. Quickstart

All operations can be managed using the helper script `./cam-sim/run.sh`:

### Step 1: Build the Container Image
Build the native ARM64 container image packaging Python 3.12, FFmpeg, and MediaMTX:
```bash
./cam-sim/run.sh build
```

### Step 2: Provide Video Files
You can either place your own recorded match MP4 files into `cam-sim/videos/` or generate synthetic test videos:
```bash
./cam-sim/run.sh generate-test-videos
```
This generates three 1080p test videos:
- `cam-sim/videos/goal_left.mp4`
- `cam-sim/videos/goal_right.mp4`
- `cam-sim/videos/center_overview.mp4`

### Step 3: Start the Simulator
Launch the container in the background exposing RTSP port `8554`:
```bash
./cam-sim/run.sh start
```

### Step 4: Inspect Logs
Follow the live startup and camera stream logs:
```bash
./cam-sim/run.sh logs
```
You will see output confirming all cameras and streams are active:
```
[INFO] cam-sim.supervisor: cam-sim is live! Simulating 3 cameras.
[INFO] cam-sim.supervisor:   * Camera 'cam1' (Underwater Goal Left):
[INFO] cam-sim.supervisor:       Low-Res:  rtsp://<host>:8554/cam1/low (640x360 @ 30fps)
[INFO] cam-sim.supervisor:       High-Res: rtsp://<host>:8554/cam1/high (1920x1080 @ 30fps)
...
```

### Step 5: Stop the Simulator
When finished, stop and clean up the container:
```bash
./cam-sim/run.sh stop
```

---

## 3. Stream Verification

### Option A: Command-Line Verification (`ffprobe`)
Verify stream resolution, codec, and framerate directly through the container:

**Check Low-Res Feed (`cam1/low`):**
```bash
podman exec cam-sim-instance ffprobe -v error \
  -show_entries stream=width,height,codec_name,r_frame_rate \
  -of default=noprint_wrappers=1 \
  rtsp://127.0.0.1:8554/cam1/low
```
*Expected Output:*
```text
codec_name=h264
width=640
height=360
r_frame_rate=30/1
```

**Check High-Res Feed (`cam1/high`):**
```bash
podman exec cam-sim-instance ffprobe -v error \
  -show_entries stream=width,height,codec_name,r_frame_rate \
  -of default=noprint_wrappers=1 \
  rtsp://127.0.0.1:8554/cam1/high
```
*Expected Output:*
```text
codec_name=h264
width=1920
height=1080
r_frame_rate=30/1
```

---

### Option B: Visual Verification (VLC Media Player)
1. Open **VLC**.
2. Select **File > Open Network...** (or press `Cmd+N`).
3. Enter the RTSP URL:
   - `rtsp://localhost:8554/cam1/high` (or `rtsp://localhost:8554/cam1/low`)
4. **Low Latency Tip for VLC**:
   - By default, VLC buffers network streams for **1000 ms (1 second)** before rendering the first frame.
   - To make video appear instantly, check **"Show more options"** in the VLC Open Network dialog and reduce **Caching** from `1000 ms` to **`150 ms`**.
   - Alternatively, open from your terminal:
     ```bash
     /Applications/VLC.app/Contents/MacOS/VLC --network-caching=150 rtsp://localhost:8554/cam1/high
     ```
5. Click **Open**. The video stream will start playing immediately.

---

### Option C: OBS Studio Verification
To add a simulated camera into OBS Studio as a scene source:
1. In OBS, create or select a Scene (e.g. `Scene - Cam 1`).
2. Add a new Source: select **Media Source**.
3. In the Media Source settings:
   - Uncheck **Local File**.
   - Set **Input**: `rtsp://localhost:8554/cam1/high`
   - Set **Network Buffering**: `1 MB` (or minimal buffering for low latency).
   - Set **Input Format**: `rtsp`
4. Click **OK**. The high-resolution simulated camera feed will now appear live in OBS.

---

### Option D: Multi-Machine / LAN Verification
If `cam-sim` is running on a secondary laptop on the pool deck:
1. Determine the simulator laptop's LAN IP address (e.g. `192.168.1.50`).
2. From the broadcast machine (or any other device on the same network), open:
   - `rtsp://192.168.1.50:8554/cam1/low`
   - `rtsp://192.168.1.50:8554/cam1/high`

---

## 4. Running Unit Tests

Unit tests validate configuration schema rules (2–6 camera limits, unique IDs, unique paths, even dimensions for H.264) and FFmpeg command generation:

```bash
# Create virtual environment and install dev dependencies
python3 -m venv cam-sim/.venv
cam-sim/.venv/bin/pip install -r cam-sim/requirements-dev.txt

# Run test suite
PYTHONPATH=cam-sim cam-sim/.venv/bin/pytest cam-sim/tests -v
```

---

## 5. Configuration Reference (`cam-sim/config.yaml`)

`cam-sim` supports **dynamic configuration hot-reloading** at runtime. When `config.yaml` is edited:
- If a camera's resolution, bitrate, fps, or source file is changed, only that camera's FFmpeg worker is restarted; other streams remain live.
- If cameras are added or removed, only the affected workers are started or stopped.
- If server port/transports or global defaults change, MediaMTX and all workers restart automatically.
- If an invalid YAML syntax or schema is saved, the error is logged and the active running configuration is preserved without crashing.

```yaml
version: "1.0"

server:
  rtsp_port: 8554                 # Port exposed to RTSP clients
  bind_address: "0.0.0.0"         # Listen on all interfaces

defaults:
  loop: true                      # Loop input video infinitely
  video_codec: "libx264"          # Video encoder
  preset: "ultrafast"             # Low-latency preset
  tune: "zerolatency"             # Zero-latency tuning
  gop_size: 15                    # Keyframe interval (2 I-frames per second at 30fps)

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
```
