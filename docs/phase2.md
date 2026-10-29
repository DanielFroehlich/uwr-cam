# Phase 2 Design Decisions: Engine Scaffolding, Configuration & OBS Client

---

## 1. Objective & Scope

Phase 2 established the core runtime foundation of the **`uwr-cam-engine`** component in its dedicated directory.

The objectives were:
1. Define a centralized, validated configuration schema for OBS and camera parameters.
2. Implement dynamic runtime hot-reloading without process termination or dropped connections.
3. Build a non-blocking, decoupled OBS Studio WebSocket v5 client with auto-reconnection and scene verification.
4. Establish the Apple Silicon hardware acceleration architectural stance.

---

## 2. Key Design Decisions

### 2.1. Dedicated Component Directory (`uwr-cam-engine/`)
- **Decision**: Place the engine in a dedicated `uwr-cam-engine/` directory symmetric to `cam-sim/`.
- **Rationale**: Isolates engine dependencies, unit test suites, and virtual environment from the camera simulator and OBS scene profiles.

### 2.2. Native macOS Host Execution for Apple Silicon Hardware Acceleration
- **Decision**: Make native execution via Python virtual environment (`.venv`) on macOS the primary and default runtime mode (`./run.sh start`), while retaining an optional `Containerfile` as a secondary CPU-only fallback.
- **Rationale**:
  - Container runtimes on macOS (Podman/Docker via Hypervisor.framework) do not support GPU passthrough for Apple Silicon Metal or CoreML/ANE. Linux guest VMs can only use CPU instructions (ARMv8 NEON).
  - In Phases 3 and 4, meeting the **100ms end-to-end latency constraint (NFR-1)** across 3 to 6 concurrent streams requires hardware-accelerated video decoding (VideoToolbox) and GPU/Neural Engine acceleration (Metal Performance Shaders / MPS).
  - Native execution gives the Python runtime direct access to `torch.device("mps")`, VideoToolbox decoders, and zero-overhead localhost networking to OBS and `cam-sim`.

### 2.3. Decoupled OBS Client Abstraction (`OBSClientProtocol`)
- **Decision**: Define a formal Python `Protocol` (`OBSClientProtocol`) and wrap `obsws-python` in `ObswsPythonClient`.
- **Rationale**:
  - Decouples the engine from the underlying third-party WebSocket SDK.
  - If blocking I/O, threading quirks, or protocol limitations ever arise, an alternative native `websockets` implementation can be dropped in without changing supervisor or switching logic.

### 2.4. Non-Blocking Asyncio Thread Delegation
- **Decision**: Wrap all synchronous network and RPC calls to `obsws_python.ReqClient` inside `asyncio.to_thread()`.
- **Rationale**:
  - The official `obsws-python` client uses blocking synchronous sockets.
  - Executing blocking socket operations on the main asyncio thread would stall the event loop, causing frame drops and jitter in downstream video processing. Threadpool delegation keeps the event loop fast and responsive.

### 2.5. Fault-Tolerant, Debounced Configuration Hot-Reloading (`ConfigWatcher`)
- **Decision**:
  - Asynchronously poll file modification time (`st_mtime_ns`) with a 0.5s debounce window.
  - If an edited YAML file has syntax errors or schema violations, catch the error, log detailed diagnostic feedback, and **retain the active running configuration without crashing**.
- **Rationale**:
  - Editors (Vim, VSCode) perform multi-step atomic writes/renames. Debouncing ensures the file has finished writing before parsing.
  - A broadcast system must never crash during a live match due to an operator typo in `config.yaml`.

### 2.6. Scene Existence Verification (FR-4.3)
- **Decision**: When connecting or reloading configuration, query OBS via `GetSceneList` and verify that every configured camera scene and fallback scene exists.
- **Rationale**: Catch missing scenes and typos immediately at startup or edit time, rather than failing during a critical live match switch.
