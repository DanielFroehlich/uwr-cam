#!/usr/bin/env bash
# ==============================================================================
# cam-sim helper script for Podman execution on macOS / Linux
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="cam-sim:latest"
CONTAINER_NAME="cam-sim-instance"
RTSP_PORT="8554"

usage() {
    echo "Usage: $0 [build|start|stop|logs|generate-test-videos|status]"
    echo ""
    echo "Commands:"
    echo "  build                 Build the cam-sim ARM64 container image"
    echo "  start                 Start cam-sim container in background"
    echo "  run                   Run cam-sim container in foreground (interactive)"
    echo "  stop                  Stop running cam-sim container"
    echo "  logs                  View live logs from cam-sim container"
    echo "  generate-test-videos  Generate synthetic MP4 video files inside container"
    echo "  status                Check container and RTSP port status"
    echo ""
    exit 1
}

cmd_build() {
    echo "==> Building ${IMAGE_NAME} with Podman..."
    podman build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Containerfile" "${SCRIPT_DIR}"
    echo "==> Build complete: ${IMAGE_NAME}"
}

cmd_start() {
    echo "==> Starting ${CONTAINER_NAME} on RTSP port ${RTSP_PORT}..."
    mkdir -p "${SCRIPT_DIR}/videos"

    # Stop any existing instance
    if podman container exists "${CONTAINER_NAME}" 2>/dev/null; then
        echo "Removing existing container ${CONTAINER_NAME}..."
        podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    podman run -d \
        --name "${CONTAINER_NAME}" \
        -p "${RTSP_PORT}:${RTSP_PORT}/tcp" \
        -p "${RTSP_PORT}:${RTSP_PORT}/udp" \
        -p 8000-8005:8000-8005/udp \
        -v "${SCRIPT_DIR}/videos:/videos:ro" \
        -v "${SCRIPT_DIR}/config.yaml:/app/config.yaml:ro" \
        "${IMAGE_NAME}"

    echo "==> ${CONTAINER_NAME} started successfully."
    echo "==> RTSP streams accessible at rtsp://localhost:${RTSP_PORT}/<camera_id>/<feed>"
}

cmd_run() {
    mkdir -p "${SCRIPT_DIR}/videos"
    if podman container exists "${CONTAINER_NAME}" 2>/dev/null; then
        podman rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "==> Running ${IMAGE_NAME} interactively..."
    podman run --rm -it \
        --name "${CONTAINER_NAME}" \
        -p "${RTSP_PORT}:${RTSP_PORT}/tcp" \
        -p "${RTSP_PORT}:${RTSP_PORT}/udp" \
        -p 8000-8005:8000-8005/udp \
        -v "${SCRIPT_DIR}/videos:/videos:ro" \
        -v "${SCRIPT_DIR}/config.yaml:/app/config.yaml:ro" \
        "${IMAGE_NAME}"
}

cmd_stop() {
    echo "==> Stopping ${CONTAINER_NAME}..."
    podman stop "${CONTAINER_NAME}" || true
    podman rm -f "${CONTAINER_NAME}" || true
    echo "==> Container stopped."
}

cmd_logs() {
    podman logs -f "${CONTAINER_NAME}"
}

cmd_generate_test_videos() {
    echo "==> Generating synthetic MP4 test videos inside container..."
    mkdir -p "${SCRIPT_DIR}/videos"

    # Run FFmpeg inside the cam-sim container to generate 3 distinct test videos
    podman run --rm \
        --entrypoint bash \
        -v "${SCRIPT_DIR}/videos:/videos:rw" \
        "${IMAGE_NAME}" \
        -c "
            echo 'Generating /videos/goal_left.mp4...' && \
            ffmpeg -y -f lavfi -i testsrc=duration=15:size=1920x1080:rate=30 \
                   -c:v libx264 -preset ultrafast -pix_fmt yuv420p /videos/goal_left.mp4 && \
            echo 'Generating /videos/goal_right.mp4...' && \
            ffmpeg -y -f lavfi -i testsrc2=duration=15:size=1920x1080:rate=30 \
                   -c:v libx264 -preset ultrafast -pix_fmt yuv420p /videos/goal_right.mp4 && \
            echo 'Generating /videos/center_overview.mp4...' && \
            ffmpeg -y -f lavfi -i smptebars=duration=15:size=1920x1080:rate=30 \
                   -c:v libx264 -preset ultrafast -pix_fmt yuv420p /videos/center_overview.mp4
        "
    echo "==> Synthetic test videos generated in ${SCRIPT_DIR}/videos/"
}

cmd_status() {
    echo "==> Container status:"
    podman ps --filter "name=${CONTAINER_NAME}"
}

ACTION="${1:-help}"
case "${ACTION}" in
    build)
        cmd_build
        ;;
    start)
        cmd_start
        ;;
    run)
        cmd_run
        ;;
    stop)
        cmd_stop
        ;;
    logs)
        cmd_logs
        ;;
    generate-test-videos)
        cmd_generate_test_videos
        ;;
    status)
        cmd_status
        ;;
    *)
        usage
        ;;
esac
