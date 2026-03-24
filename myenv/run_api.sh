#!/bin/bash
# Script chạy API nhanh

echo "🚀 Khởi động Pill Recognition API..."

# Cách 1: Chạy với uvicorn (nhanh nhất, có auto-reload khi dev)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Cách 2: Chạy trực tiếp (nếu không cần auto-reload)
# python -m api.main

# Cách 3: Chạy với nhiều workers (cho production)
# uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4