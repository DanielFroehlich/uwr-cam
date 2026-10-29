# Phase 4 Design Decisions: Ball Detection & Switching Arbitration

---

## 1. Objective & Scope

Phase 4 completed the intelligent core of the **`uwr-cam-engine`** by integrating ball detection, composite camera scoring, and automated scene switching arbitration.

The objectives were:
1. Provide a dataset extraction tool to bootstrap YOLO training on match footage.
2. Abstract the deep-learning ball detection engine for PyTorch/Ultralytics with Apple Silicon Metal/MPS hardware acceleration.
3. Combine ball tracking ($B_i$) and motion density ($M_i$) into a unified composite camera score ($S_i$).
4. Implement an intelligent state machine (`SwitchingArbiter`) that drives OBS Studio scene switches while eliminating flapping, respecting dwell times, and providing fallback and manual override.

---

## 2. Key Design Decisions

### 2.1. Training Dataset Frame Extractor (`scripts/extract_frames.py`)
- **Decision**: Provide an automated frame extraction script sampling recorded match videos (`cam-sim/videos/*.mp4`) at a configurable interval (default: 1 FPS) and exporting full-resolution JPEGs to `datasets/raw/`.
- **Rationale**:
  - Training a YOLO model requires high-quality annotated images of the white underwater ball amidst scrums, divers, and water caustics.
  - Extracting at 1 FPS from 3 cameras produces ~180 diverse training frames per minute of video, avoiding redundant consecutive frames.
  - The extracted frames can be uploaded directly to annotation platforms (Roboflow, CVAT).

### 2.2. Deep Learning Ball Detector (`BallDetectorYOLO` / YOLO-World)
- **Decision**: Integrate Ultralytics `YOLO-World` (`yolov8s-worldv2.pt`) using open-vocabulary prompts:
  `["underwater ball", "white ball"]` with hardware acceleration on Apple Silicon Metal Performance Shaders (`device="mps"`).
- **Validation Results**:
  - Zero training required: recognizes the underwater rugby ball submerged in pool water using pre-trained vision-language embeddings.
  - Achieved **0.73–0.80 confidence** on live match footage.
  - Sub-15ms inference latency per frame on Apple Silicon GPU (`mps`), well within the 100ms latency budget (NFR-1).
  - Designed with singleton model sharing: a single `BallDetectorYOLO` instance is shared across all camera pipelines to eliminate redundant GPU VRAM overhead.
  - Extensible: also accepts custom trained `.pt` weight files when specified via `config.yaml`.

### 2.3. Composite Camera Scoring Formula (FR-2.3)
- **Decision**: Calculate the composite score per camera $i$ inside `PipelineManager`:
  $$S_i = (w_{\text{ball}} \cdot B_i) + (w_{\text{motion}} \cdot M_i)$$
- **Rationale**:
  - Ball location is the primary indicator of action ($w_{\text{ball}} = 0.7$).
  - When the ball is momentarily obscured beneath players during a goal scrum, high motion density ($w_{\text{motion}} = 0.3$) keeps the broadcast focused on the goal action rather than reverting prematurely.
  - Weights are configurable per camera in `config.yaml`.

### 2.4. Switching Arbiter State Machine (`SwitchingArbiter`)
- **Decision**: Implement a decoupled state machine enforcing broadcast switching rules:
  1. **Manual Override (FR-3.3)**: If `manual_override: true`, all automated switching commands are suppressed, granting the human broadcast director sole control in OBS.
  2. **Anti-Flapping Dwell Time (FR-3.1)**: If `time - last_switch_time < min_dwell_time_sec` (e.g., 2.5s), hold the current camera regardless of score changes. This prevents rapid toggling that causes viewer fatigue.
  3. **Hysteresis Margin (FR-3.1)**: A candidate camera $C_{\text{max}}$ must exceed the *currently active camera's score* by at least `hysteresis_threshold` (e.g., $+0.15$):
     $$S_{\text{max}} > S_{\text{current}} + \Delta_{\text{hysteresis}}$$
     This prevents flapping between two cameras observing similar activity levels.
  4. **Inactivity Fallback Timeout (FR-3.2)**: If all cameras remain below `inactivity_threshold: 0.05` for `fallback_timeout_sec` (e.g., 5.0s), automatically switch to the designated `fallback_scene` (e.g., `base`).

### 2.5. Log Level Separation
- **Decision**:
  - `INFO`: Reserved for high-signal events: switching decisions (`Switching triggered: Camera 'cam1' score 0.30 exceeds required 0.15...`) and periodic aggregated status summaries (`[Status] Active: 'cam1' | Cameras: ...`).
  - `DEBUG`: Reserved for low-level mechanics: `Arbiter commanded scene switch to: '...'` and `OBS Program scene switched to: '...'`.
- **Rationale**: Keeps live broadcast monitoring clear and readable without duplicate switch confirmations.
