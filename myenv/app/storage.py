"""
Module lưu trữ dữ liệu predictions cho analytics.

Mô tả:
    - Lưu thông tin predictions vào JSON file
    - Load lịch sử predictions
    - Xóa dữ liệu cũ (nếu cần)
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# ========================================
# Constants
# ========================================
DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "predictions.json"


# ========================================
# Ensure data directory exists
# ========================================
def ensure_data_directory():
    """Tạo data directory nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ========================================
# Save Prediction
# ========================================
def save_prediction(
    predicted_pill: str,
    confidence: float,
    probabilities: Dict[str, float],
    filename: str = None,
) -> bool:
    """
    Lưu prediction vào file.

    Args:
        predicted_pill: Tên viên thuốc nhận diện
        confidence: Độ tin cậy (0-1)
        probabilities: Dict xác suất cho các lớp
        filename: Tên file ảnh (optional)

    Returns:
        True nếu lưu thành công, False nếu thất bại
    """
    try:
        ensure_data_directory()

        # Tạo record prediction
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "predicted_pill": predicted_pill,
            "confidence": float(confidence),
            "probabilities": {k: float(v) for k, v in probabilities.items()},
            "filename": filename or "unknown",
        }

        # Load existing predictions
        predictions = load_all_predictions()

        # Add new prediction
        predictions.append(prediction_record)

