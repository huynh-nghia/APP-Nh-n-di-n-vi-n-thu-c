#!/bin/bash
# Script chạy API nhanh

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

HOST="${BACKEND_HOST:-0.0.0.0}"
PORT="${BACKEND_PORT:-8000}"
ENV_NAME="${ENVIRONMENT:-development}"
MODEL_PATH="${ML_CLASSIFIER_MODEL_PATH:-models/ml_classifier.pkl}"

# Đảm bảo logger có thư mục writable; nếu logs/ không ghi được thì fallback sang /tmp
LOG_DIR_DEFAULT="$SCRIPT_DIR/logs"
LOG_DIR_FALLBACK="/tmp/pill-recognition-logs"

mkdir -p "$LOG_DIR_DEFAULT" 2>/dev/null || true
if [ -w "$LOG_DIR_DEFAULT" ]; then
  export LOG_DIR="$LOG_DIR_DEFAULT"
else
  mkdir -p "$LOG_DIR_FALLBACK"
  export LOG_DIR="$LOG_DIR_FALLBACK"
fi

echo "🚀 Khởi động Pill Recognition API..."
echo "- Thư mục làm việc: $SCRIPT_DIR"
echo "- Môi trường: $ENV_NAME"
echo "- Host/Port: $HOST:$PORT"
echo "- LOG_DIR: $LOG_DIR"

if [ -f "$MODEL_PATH" ]; then
  echo "- Model: $MODEL_PATH"
else
  echo "⚠️ Chưa tìm thấy model tại: $MODEL_PATH"
fi

if [ "$ENV_NAME" = "production" ]; then
  exec uvicorn api.main:app --host "$HOST" --port "$PORT"
else
  exec uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
fi