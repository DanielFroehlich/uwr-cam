# uwr-cam-engine: Underwater Rugby Automated Camera Switcher Engine
## Component Design Specification (Phase 2)

---

### 1. Overview & Purpose

`uwr-cam-engine` is the central analysis and switching controller of the **uwr-cam** automated production system.

In Phase 2, `uwr-cam-engine` establishes:
1. **Dynamic Configuration Management**: A validated YAML configuration parser with automated runtime hot-reloading (without process restarts).
2. **OBS Studio WebSocket v5 Client**: A decoupled, non-blocking scene-switching client connecting to OBS Studio via WebSocket v5 RPC, verifying scenes, and providing auto-reconnect resilience.
3. **Apple Silicon Hardware-First Architecture**: Configured for native host execution on macOS Apple Silicon to guarantee access to **Metal Performance Shaders (MPS)**, **Apple Neural Engine (ANE)**, and **VideoToolbox** hardware video decoders for Phases 3 and 4 computer vision models.

---

### 2. Architecture & Data Flow

```
                                  +------------------------------+
                                  | uwr-cam-engine/config.yaml   |
                                  +--------------+---------------+
                                                 |
                                                 v (File change event)
                                  +------------------------------+
                                  |        ConfigWatcher         |
                                  |  - Debounced mtime polling   |
                                  |  - Pydantic schema validator |
                                  +--------------+---------------+
                                                 | (On valid update)
                                                 v
+-----------------------------------------------------------------------------------------------+
|                                      EngineSupervisor                                         |
|                                                                                               |
|  +---------------------------+    asyncio     +--------------------------------------------+  |
|  |     Health Monitor        | <------------> |            OBSClientProtocol               |  |
|  | - Auto-reconnect on drop  |                |         (Abstract OBS Interface)           |  |
|  | - Re-verify scene states  |                +---------------------+----------------------+  |
|  +---------------------------+                                      |                         |
|                                                                     v                         |
|                                               +--------------------------------------------+  |
|                                               |            ObswsPythonClient               |  |
|                                               | - Underlying: obsws-python ReqClient       |  |
|                                               | - Thread delegation: asyncio.to_thread     |  |
|                                               | - Non-blocking asyncio event loop          |  |
|                                               +---------------------+----------------------+  |
+---------------------------------------------------------------------|-------------------------+
                                                                      |
                                                                      | WebSocket RPC (Port 4455)
                                                                      | SetCurrentProgramScene
                                                                      v
                                                        +-----------------------------+
                                                        |         OBS Studio          |
                                                        |  - Program scene switching  |
                                                        |  - Live broadcast output    |
                                                        +-----------------------------+
```

---

### 3. Key Design Decisions

#### 3.1. Native macOS Execution vs. Containerization (Apple Silicon MPS/GPU)
- **Decision**: Make native macOS execution via Python virtual environment (`.venv`) the primary and default runtime mode for `uwr-cam-engine`.
- **Rationale**:
  - Container runtimes on macOS (Podman/Docker via Linux VMs) run within a virtualized guest kernel without Metal GPU passthrough. Only CPU instructions (ARMv8 NEON) are available inside containers.
  - In Phase 3 (optical flow motion analysis) and Phase 4 (YOLO/PyTorch UWR ball tracking), computer vision inference requires sub-10ms latency to comply with our **100ms end-to-end latency constraint (NFR-1)**.
  - Running natively on macOS provides full access to:
    - **Metal Performance Shaders (MPS)** via `torch.device("mps")`
    - **Apple Neural Engine (ANE)** via CoreML
    - Hardware-accelerated H.264 video decoding via **VideoToolbox**
    - Zero container networking overhead when connecting to local OBS (`localhost:4455`) and `cam-sim` (`localhost:8554`).
  - An optional `Containerfile` is preserved for headless CPU server deployments if needed.

#### 3.2. Decoupled OBS Client Abstraction (`OBSClientProtocol`)
- **Decision**: Define a formal Python `Protocol` (`OBSClientProtocol`) implemented by `ObswsPythonClient`.
- **Rationale**:
  - Encapsulates the third-party `obsws-python` library behind a clean interface.
  - If blocking I/O or protocol limitations arise in later phases, an alternative native `websockets` implementation can be dropped in without touching supervisor, detection, or switching code.

#### 3.3. Non-Blocking Asyncio Thread Delegation
- **Decision**: Wrap all blocking `obsws_python.ReqClient` network calls in `asyncio.to_thread()`.
- **Rationale**:
  - `obsws-python` is synchronous. Executing synchronous sockets on the main asyncio event loop would introduce jitter and frame drops into real-time video grabbers.
  - Delegating to Python's background thread pool ensures the asyncio loop runs unhindered.

#### 3.4. Debounced, Fault-Tolerant Hot-Reloading (`ConfigWatcher`)
- **Decision**: Poll file modification times asynchronously with configurable debounce intervals (0.5s).
- **Rationale**:
  - Editors (Vim, VSCode) often perform multiple write/rename cycles when saving files. Debouncing prevents reloading partially-written YAML files.
  - If a saved YAML file violates schema validation or syntax rules, `ConfigWatcher` catches the error, logs a detailed warning, and **retains the active running configuration**, ensuring continuous broadcast operation without crashes.

#### 3.5. Scene Existence Verification (FR-4.3)
- **Decision**: Upon connecting or re-loading configuration, compare camera scene mappings and the fallback scene against `get_scenes()` from OBS.
- **Rationale**:
  - Detects typos in scene names early before attempting switching during a live game.

#### 3.6. Director Manual Override via OBS Scene Selection (FR-3.3)
- **Decision**: Implement dual-stage scene verification:
  1. **Periodic Scene Synchronization (`_obs_scene_sync_loop`)**: Background task polling OBS every `scene_poll_interval_sec` (default 0.5s) to detect manual director scene selections without jittering the CV processing loop.
  2. **Pre-Switch Verification (`_cv_processing_loop`)**: Immediately before calling `obs_client.set_scene()`, queries the active OBS program scene. If an unmanaged manual scene (e.g. commentary, replay, interview, sponsor) is active, or if the scene changed since evaluation, the automated switch is aborted and the arbiter enters manual override mode.
  3. **Seamless Resumption**: As soon as the human director selects any automated scene (any configured camera feed or the fallback scene), automated switching automatically resumes and enforces minimum dwell time on the selected scene before transitioning.

#### 3.7. Live Technical Metrics Overlay in OBS
- **Decision**: Update a configurable OBS text source in real-time via `set_input_settings(overlay=True)` throttled at `update_interval_sec`.
- **Rationale**:
  - Automatically verifies and creates the text input in OBS (`text_ft2_source_v2` on macOS/Linux or `text_gdiplus_v2` on Windows) at startup or reconnection.
  - Dynamically generates multi-line metrics (ball confidence, motion score, composite score, and active scene) formatted across all configured camera feeds.
  - Gives the broadcast director instant visual feedback on automated camera rankings with zero disk I/O.

#### 3.8. Live Ball Marker Bounding Box Overlay
- **Decision**: Render a dynamic bounding box over the detected underwater rugby ball in OBS using a native `image_source` scene item positioned via `set_scene_item_transform`.
- **Rationale**:
  - Dynamically queries the OBS profile's base canvas resolution (`get_video_settings().base_width`, `base_height`) to accurately scale coordinates from low-res model frames to any broadcast resolution (1080p, 4K, etc.).
  - Auto-generates a clean, transparent hollow neon border PNG asset locally on startup.
  - Controls marker visibility via `set_scene_item_enabled`: enabled when the ball is detected on the currently active camera feed, and hidden when lost, obscured, or when switching away.
  - Supports live runtime toggling and re-configuration via config hot-reloading (`on_config_reload`).
