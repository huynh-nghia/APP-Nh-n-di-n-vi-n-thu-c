#!/bin/bash
# Script chạy UI nhanh

echo "🚀 Khởi động Pill Recognition UI..."

# Chạy Streamlit UI
streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0