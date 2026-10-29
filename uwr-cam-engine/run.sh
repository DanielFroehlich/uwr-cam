#!/usr/bin/env bash
# ==============================================================================
# uwr-cam-engine helper script for macOS Apple Silicon
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
IMAGE_NAME="uwr-cam-engine:latest"
CONTAINER_NAME="uwr-cam-engine-instance"

usage() {
    echo "Usage: $0 [start|dev|test|status|container-build|container-run]"
    echo ""
    echo "Native macOS Host Commands (Apple Silicon GPU / MPS / Neural Engine):"
    echo "  start       Run engine directly on macOS host via virtualenv (Primary)"
    echo "  test        Run automated test suite with pytest"
    echo "  status      Check OBS and cam-sim connectivity"
    echo ""
    echo "Optional Container Commands (CPU fallback only):"
    echo "  container-build  Build optional Linux ARM64 container image"
    echo "  container-run    Run containerized CPU version"
    echo ""
    exit 1
}

ensure_venv() {
    if [ ! -d "${VENV_DIR}" ]; then
        echo "==> Creating Python virtual environment in ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
        "${VENV_DIR}/bin/pip" install --upgrade pip
        "${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements-dev.txt"
    fi
}

cmd_start() {
    ensure_venv
    echo "==> Starting uwr-cam-engine natively on macOS (with Apple Silicon Metal/MPS support)..."
    export PYTHONPATH="${SCRIPT_DIR}"
    exec "${VENV_DIR}/bin/python" -m src.main -c "${SCRIPT_DIR}/config.yaml"
}

cmd_test() {
    ensure_venv
    echo "==> Running uwr-cam-engine test suite..."
    PYTHONPATH="${SCRIPT_DIR}" "${VENV_DIR}/bin/pytest" -v "${SCRIPT_DIR}/tests"
}

cmd_status() {
    echo "=== uwr-cam-engine Status Check ==="
    echo "Checking OBS WebSocket (port 4455):"
    if nc -z localhost 4455 2>/dev/null; then
        echo "  [OK] OBS WebSocket is listening on port 4455"
    else
        echo "  [FAIL] OBS WebSocket is NOT detected on port 4455"
    fi

    echo "Checking cam-sim RTSP Server (port 8554):"
    if nc -z localhost 8554 2>/dev/null; then
        echo "  [OK] cam-sim RTSP Server is listening on port 8554"
    else
        echo "  [FAIL] cam-sim RTSP Server is NOT detected on port 8554"
    fi
}

cmd_container_build() {
    echo "==> Building ${IMAGE_NAME} container with Podman (CPU only)..."
    podman build -t "${IMAGE_NAME}" -f "${SCRIPT_DIR}/Containerfile" "${SCRIPT_DIR}"
}

cmd_container_run() {
    echo "==> Running ${IMAGE_NAME} container interactively..."
    podman run --rm -it \
        --name "${CONTAINER_NAME}" \
        --network host \
        -v "${SCRIPT_DIR}/config.yaml:/app/config.yaml:ro" \
        "${IMAGE_NAME}"
}

COMMAND="${1:-start}"
case "${COMMAND}" in
    start|dev|run)
        cmd_start
        ;;
    test)
        cmd_test
        ;;
    status)
        cmd_status
        ;;
    container-build)
        cmd_container_build
        ;;
    container-run)
        cmd_container_run
        ;;
    *)
        usage
        ;;
esac
