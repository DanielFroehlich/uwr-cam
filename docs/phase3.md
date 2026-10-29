# Phase 3 Design Decisions: Video Ingestion & Motion Detection

---

## 1. Objective & Scope

Phase 3 implemented the real-time video ingestion pipeline and the secondary computer vision metric: **Motion Density Analysis (FR-2.2)**.

The objectives were:
1. Ingest low-resolution RTSP streams concurrently from 2 to 6 cameras.
2. Eliminate buffering latency to satisfy the sub-100ms latency budget (FR-1.4).
3. Compute motion density in underwater rugby environments where water ripples create persistent background noise.
4. Establish multi-tier observability (trace/debug vs. periodic summary info).

---

## 2. Key Design Decisions

### 2.1. Dedicated Threaded Ingestion with I/O-Paced Frame Dropping (`FrameGrabber`)
- **Decision**: Run a dedicated background thread per camera running an continuous loop around `cv2.VideoCapture.grab()`. Decode (`retrieve()`) only when the main CV loop polls for the latest frame.
- **CPU Efficiency Analysis**:
  - `cap.grab()` on an RTSP stream (via FFmpeg) uses **blocking network I/O**. It suspends the thread while waiting for the next packet from the RTSP server.
  - The thread naturally paces itself to the stream's framerate (~30 FPS = ~33ms block time per grab), avoiding high CPU consumption across up to 6 cameras.
  - In error paths (disconnection/stream loss), safety back-off sleeps (`reconnect_interval_sec`) prevent CPU spinning.
- **Zero-Latency Buffering (FR-1.4)**:
  - Standard OpenCV `VideoCapture.read()` buffers frames in an internal FIFO queue. If CV processing takes longer than one frame interval, latency compounds and the stream lags behind real-time by seconds.
  - Aggressive `grab()` continuously drains and updates `_latest_frame`, completely bypassing the FIFO queue. When the CV pipeline polls `read_latest()`, it receives the absolute newest frame.

### 2.2. Adaptive Background Subtraction (`MotionDetectorMOG2`)
- **Decision**: Use OpenCV's Mixture of Gaussians (MOG2) algorithm (`createBackgroundSubtractorMOG2`) rather than simple frame differencing or dense optical flow.
- **Rationale**:
  - *Why not Frame Differencing?* Surface water ripples and fluctuating caustics create continuous pixel variations that falsely register as massive motion across the entire frame.
  - *Why not Dense Optical Flow?* Farneback optical flow is computationally heavy across 3–6 streams and computes vectors for ripples that require expensive post-filtering.
  - *Why MOG2?* MOG2 dynamically models the background probability distribution per pixel over time (`history=150`). It "learns" repetitive ripple motion as background, effectively ignoring it, and segments out only the sudden, high-contrast foreground movements of underwater rugby players.

### 2.3. Noise Filtering & Normalization
- **Decision**:
  - Morphological opening with a $3 \times 3$ elliptical structuring element to eliminate salt-and-pepper noise.
  - External contour filtering discarding any connected components smaller than `min_contour_area: 300` pixels.
  - Occupancy scaling: In underwater rugby footage, players taking up 15% of the frame represents an intense goal scrum. The score is normalized linearly such that $15\%$ frame occupancy yields $1.0$ (capped at $1.0$).

### 2.4. Interchangeable Protocol Contract (`MotionDetectorProtocol`)
- **Decision**: Define a formal Python `Protocol` for the motion detector.
- **Rationale**: Keeps the implementation modular and named `MotionDetectorMOG2`, allowing seamless benchmarking or swapping with alternative detectors (e.g. KNN or optical flow) without touching the pipeline manager.

### 2.5. Multi-Tier Observability & Periodic Status Logging
- **Decision**:
  - `DEBUG`: Logs granular per-frame processing times, raw scores, and network events.
  - `INFO`: Suppresses per-frame noise and instead emits a periodic, aggregated summary of all camera scores and the active OBS scene every `status_interval_sec` seconds (configurable, default: 5.0s).
- **Rationale**: Avoids terminal spamming at 30 FPS while maintaining high operational awareness for the broadcast director.
