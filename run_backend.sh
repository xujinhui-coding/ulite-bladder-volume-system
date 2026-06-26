#!/usr/bin/env bash
# 启动封装 — 单进程 Uvicorn 服务
# 用法: bash run_backend.sh [--port 8000]

set -euo pipefail

PORT="${2:-8000}"

export UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
export UVICORN_PORT="${PORT}"

cd "$(dirname "$0")"

echo "[U-Lite Backend] Starting on ${UVICORN_HOST}:${UVICORN_PORT} (workers=1)"
exec python -m uvicorn main:app \
    --host "${UVICORN_HOST}" \
    --port "${UVICORN_PORT}" \
    --workers 1 \
    --no-access-log
