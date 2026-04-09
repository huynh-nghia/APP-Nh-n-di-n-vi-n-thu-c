#!/bin/bash
# Script chạy UI nhanh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load biến môi trường từ file .env nếu có
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

ENV_NAME="${ENVIRONMENT:-development}"
DEFAULT_UI_PORT="8501"
if [ "$ENV_NAME" = "production" ]; then
  DEFAULT_UI_PORT="7860"
fi

UI_HOST="${STREAMLIT_HOST:-0.0.0.0}"
UI_PORT="${STREAMLIT_PORT:-$DEFAULT_UI_PORT}"
BACKEND_HOST_VALUE="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT_VALUE="${BACKEND_PORT:-8000}"

# Nếu chưa có BACKEND_URL thì tự suy ra từ BACKEND_HOST/BACKEND_PORT
if [ -z "${BACKEND_URL:-}" ]; then
  if [ "$BACKEND_HOST_VALUE" = "0.0.0.0" ]; then
    export BACKEND_URL="http://127.0.0.1:${BACKEND_PORT_VALUE}"
  else
    export BACKEND_URL="http://${BACKEND_HOST_VALUE}:${BACKEND_PORT_VALUE}"
  fi
fi

echo "🚀 Khởi động Pill Recognition UI..."
echo "- Thư mục làm việc: $SCRIPT_DIR"
echo "- Môi trường: $ENV_NAME"
echo "- UI: http://$UI_HOST:$UI_PORT"
echo "- Backend URL: $BACKEND_URL"

STREAMLIT_ARGS=(
  run app/ui.py
  --server.port "$UI_PORT"
  --server.address "$UI_HOST"
  --server.headless true
  --browser.gatherUsageStats false
)

if [ "$ENV_NAME" = "production" ]; then
  STREAMLIT_ARGS+=(
    --server.enableCORS false
    --server.enableXsrfProtection false
  )
fi

exec streamlit "${STREAMLIT_ARGS[@]}"