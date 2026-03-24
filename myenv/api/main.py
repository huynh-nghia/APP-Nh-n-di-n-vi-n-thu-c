"""
File này tạo REST API cho Pill Recognition App.

Mục đích:
    - Nhận file ảnh upload từ người dùng
    - Xác thực API Key
    - Trích xuất features (DL) + Phân loại (ML)
    - Trả về kết quả nhận diện viên thuốc

Kiến trúc:
    Client → FastAPI → Feature Extractor (DL) → Classifier (ML) → Kết quả
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import từ src modules
try:
    from src.logger import ghi_log, ghi_loi, ghi_bao_mat
    from src.security import kiem_tra_api_key, kiem_tra_file_upload
    from src.feature_extractor import FeatureExtractor
    from src.classifier import PillClassifier
    from src.pill_label_utils import (
        chuyen_probabilities_sang_ten,
        format_hien_thi,
        load_mapping,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.logger import ghi_log, ghi_loi, ghi_bao_mat
    from src.security import kiem_tra_api_key, kiem_tra_file_upload
    from src.feature_extractor import FeatureExtractor
    from src.classifier import PillClassifier
    from src.pill_label_utils import (
        chuyen_probabilities_sang_ten,
        format_hien_thi,
        load_mapping,
    )

# ========================================
# Global Models
# ========================================
feature_extractor = None
classifier = None


# ========================================
# Response Models
# ========================================

class PredictionResult(BaseModel):
    """Kết quả dự đoán."""
    success: bool
    message: str
    predicted_pill: Optional[str] = None
    predicted_pill_display: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[dict] = None
    display_probabilities: Optional[dict] = None
    error: Optional[str] = None


class HealthCheckResult(BaseModel):
    """Kết quả health check."""
    status: str
    models_loaded: bool
    device: str


# ========================================
# Lifespan
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý startup và shutdown."""
    global feature_extractor, classifier

    # Startup
    try:
        ghi_log("=== Khởi động Pill Recognition API ===")

        # Tải Feature Extractor
        ghi_log("Đang tải Feature Extractor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_extractor = FeatureExtractor(ten_model="mobilenetv2", thiet_bi=device)
        ghi_log(f"Feature Extractor đã sẵn sàng | Thiết bị: {device}")

        # Tải Classifier
        ghi_log("Đang tải ML Classifier...")
        classifier = PillClassifier(loai="svm")

        # Tải model từ file (nếu có)
        model_path = os.getenv("ML_CLASSIFIER_MODEL_PATH", "models/ml_classifier.pkl")
        classifier.tai_model(model_path)

        ghi_log("=== API đã sẵn sàng ===")

    except Exception as e:
        ghi_loi(f"Không thể khởi động API: {str(e)}", e)
        raise

    yield

    # Shutdown
    ghi_log("=== Đang tắt API ===")


# ========================================
# FastAPI App
# ========================================

app = FastAPI(
    title="Pill Recognition API",
    description="API nhận diện viên thuốc từ ảnh",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# Endpoints
# ========================================

@app.get("/health", response_model=HealthCheckResult)
async def health_check():
    """Kiểm tra trạng thái API."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_loaded = feature_extractor is not None and classifier is not None

        ghi_log("Health check được thực hiện")

        return HealthCheckResult(
            status="healthy",
            models_loaded=models_loaded,
            device=device,
        )
    except Exception as e:
        ghi_loi(f"Health check thất bại: {str(e)}", e)
        raise HTTPException(status_code=500, detail="Health check thất bại")


@app.post("/predict", response_model=PredictionResult)
async def predict_pill(
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """
    Nhận diện viên thuốc từ ảnh upload.
    
    Args:
        file: File ảnh (JPEG/PNG, max 5MB)
        x_api_key: API Key (header)
    
    Returns:
        Kết quả nhận diện
    """
    try:
        # Xác thực API Key
        try:
            kiem_tra_api_key(x_api_key)
            ghi_log("API Key hợp lệ")
        except HTTPException:
            ghi_bao_mat(f"Truy cập trái phép | API Key: {x_api_key[:10] if x_api_key else 'None'}...")
            raise

        # Kiểm tra tên file
        if not file.filename:
            ghi_bao_mat("Tên file rỗng")
            raise HTTPException(status_code=400, detail="Tên file là bắt buộc")

        ghi_log(f"Đang xử lý | File: {file.filename}")

        # Đọc file content
        await file.seek(0)
        noi_dung = await file.read()

        # Nếu rỗng, thử đọc lại
        if not noi_dung:
            try:
                file.file.seek(0)
                noi_dung = file.file.read()
                ghi_log(f"Đọc fallback thành công | Kích thước: {len(noi_dung)} bytes")
            except Exception as e:
                ghi_loi(f"Đọc fallback thất bại: {str(e)}", e)

        # Kiểm tra file rỗng
        if not noi_dung or len(noi_dung) == 0:
            ghi_bao_mat("Nội dung file rỗng")
            raise HTTPException(status_code=400, detail="File upload rỗng")

        # Kiểm tra file (tên, kích thước, định dạng, MIME)
        kiem_tra_file_upload(file.filename, noi_dung)

        ghi_log(f"File hợp lệ | Kích thước: {len(noi_dung)} bytes")

        # Trích xuất features
        features = feature_extractor.trich_xuat_tu_anh(noi_dung)
        ghi_log(f"Đã trích xuất features | Shape: {features.shape}")

        # Phân loại
        ket_qua = classifier.du_doan_mot_anh(features)

        # Format kết quả
        mapping = load_mapping()
        predicted_display = format_hien_thi(ket_qua["predicted_label"], mapping)
        display_probs = chuyen_probabilities_sang_ten(ket_qua["probabilities"], mapping)

        ghi_log(f"Dự đoán thành công | {predicted_display} | Confidence: {ket_qua['confidence']:.4f}")

        return PredictionResult(
            success=True,
            message="Dự đoán thành công",
            predicted_pill=ket_qua["predicted_label"],
            predicted_pill_display=predicted_display,
            confidence=ket_qua["confidence"],
            probabilities=ket_qua["probabilities"],
            display_probabilities=display_probs,
        )

    except HTTPException:
        raise
    except Exception as e:
        ghi_loi(f"Lỗi dự đoán: {str(e)}", e)
        return PredictionResult(
            success=False,
            message="Dự đoán thất bại",
            error=str(e),
        )


@app.get("/")
async def root():
    """Thông tin API."""
    return {
        "message": "Pill Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict (cần X-API-Key)",
        },
    }


# ========================================
# Exception Handler
# ========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Xử lý HTTP exception."""
    ghi_bao_mat(f"[HTTP {exc.status_code}] {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": "Lỗi",
            "error": exc.detail,
        },
    )


# ========================================
# Entry Point
# ========================================

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", 8000))

    ghi_log(f"Khởi động server trên {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
    )