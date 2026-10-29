# Underwater Rugby Automated Camera Switcher (`uwr-cam`)

[![Vibe Coded with Gemini](https://img.shields.io/badge/vibe%20coded%20with-Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://gemini.google.com/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg?style=for-the-badge)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black?style=for-the-badge&logo=apple)](https://apple.com/)
[![Container: Podman](https://img.shields.io/badge/Container-Podman-purple?style=for-the-badge&logo=podman)](https://podman.io/)

---

## 🏊 Overview & Objective

**`uwr-cam`** is an automated, real-time video stream analyzer and production switcher controller specifically tailored for **Underwater Rugby (UWR)** broadcasts.

The system ingests video feeds from multiple IP cameras (2 to 6 cameras), analyzes each feed using computer vision to track the UWR ball and player motion density, and dynamically commands **OBS Studio** via WebSocket to switch the live program scene to the camera showing the most active play or the ball.

### Dual-Stream Client-Pull Architecture

```
                          +----------------------------------------------+
                          |        IP Cameras / cam-sim (2 to 6 cams)    |
                          |            RTSP Streaming Servers            |
                          +----------------------+-----------------------+
                                                 |
                       +-------------------------+-------------------------+
                       | Low-Res (RTSP / RTP)                              | High-Res (1K/4K RTSP / RTP)
                       v                                                   v
        +-----------------------------+                     +-----------------------------+
        |      uwr-cam Engine         |                     |         OBS Studio          |
        |                             |                     |                             |
        |  +-----------------------+  |                     |  +-----------------------+  |
        |  | Stream Ingestion &    |  |                     |  | High-Res Inputs       |  |
        |  | Frame Grabber Manager |  |                     |  | (1 Scene per Camera)  |  |
        |  +-----------+-----------+  |                     |  +-----------^-----------+  |
        |              |              |                     |              |              |
        |              v              |                     |              |              |
        |  +-----------------------+  |                     |              |              |
        |  | CV Detection Engine:  |  |                     |              |              |
        |  | - Ball Detection      |  |                     |              |              |
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

## ⚡ Key Requirements

- **Latency Constraint (NFR-1)**: Maximum end-to-end latency of **100 ms** from frame capture to OBS switching command dispatch.
- **RTSP Transports**: 
  - **RTP over UDP** for production on dedicated wired Ethernet cable networks to eliminate TCP retransmission latency spikes.
  - **RTP over TCP** for local containerized development and NAT traversal.
- **Dynamic Reconfiguration**: Central YAML configuration applied at runtime with file watching without restarting the primary process.
- **Platform**: macOS on Apple Silicon (`arm64`), containerized via Podman.

---

## 📦 Project Structure & Components

| Component | Directory | Status | Description |
| :--- | :--- | :--- | :--- |
| **`cam-sim`** | [`cam-sim/`](cam-sim/) | **Completed** | Multi-camera RTSP simulator reading MP4 match files and streaming dual-resolution feeds in an endless loop via embedded MediaMTX and FFmpeg. |
| **`uwr-cam`** | `src/` | *In Planning* | Central computer vision detection engine, stream frame grabber, arbitration logic, and OBS WebSocket v5 controller. |
| **Specifications** | [`REQUIREMENTS.md`](REQUIREMENTS.md) | **Active** | System specification, architectural decisions, and functional requirements. |

---

## 🚀 Quickstart: Camera Simulator (`cam-sim`)

To test streaming without physical IP cameras, use the built-in simulator:

```bash
# 1. Build ARM64 container image with Podman
./cam-sim/run.sh build

# 2. Generate 1080p synthetic test videos (or put your own MP4s in cam-sim/videos/)
./cam-sim/run.sh generate-test-videos

# 3. Start the simulator exposing RTSP port 8554
./cam-sim/run.sh start

# 4. Stream in VLC (instant lock-on):
vlc --network-caching=150 rtsp://localhost:8554/cam1/high
```

See [`cam-sim/README.md`](cam-sim/README.md) for complete details.

---
🤖 AI Development & Provenance

> **This project is 100% vibe-coded with [Gemini CLI](https://gemini.google.com/).**  
> All architecture specifications, component designs, Python implementations, FFmpeg transcoding filtergraphs, MediaMTX streaming configurations, containerfiles, orchestration scripts, and automated test suites are authored interactively with Gemini acting as the software engineering agent under human direction.


## 📄 License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
