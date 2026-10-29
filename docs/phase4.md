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

### 2.2. Deep Learning Ball Detector Abstraction (`BallDetectorYOLO`)
- **Decision**: Define a formal `BallDetectorProtocol` and implement `BallDetectorYOLO` as an inference engine prepared for PyTorch/Ultralytics models on Apple Silicon MPS (`device="mps"`), functioning as an extensible stub until trained model weights are provided.
- **Rationale**:
  - *Why not Heuristics / HSV Color Filtering?* Heuristic color segmentation fails when white swim caps, referee markers, or surface light reflections enter the frame. Deep learning (YOLOv8/v11) learns spatial, texture, and shape contexts necessary to distinguish the ball in scrums.
  - Using an inference stub allowed the full end-to-end switching and arbitration pipeline to be implemented and empirically verified without blocking on model training.

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
