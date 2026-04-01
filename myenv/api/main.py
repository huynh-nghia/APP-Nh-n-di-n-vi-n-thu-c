"""
Pill Recognition API - Backend

API này giúp nhận diện viên thuốc từ ảnh người dùng upload.
Cách hoạt động:
1. Người dùng gửi ảnh lên API
2. API kiểm tra API Key (bảo mật)
3. Trích xuất đặc điểm ảnh bằng AI (Deep Learning)
4. Phân loại viên thuốc bằng Machine Learning
5. Trả kết quả về cho người dùng

Các endpoint chính:
- GET /health: Kiểm tra API có hoạt động không
- POST /predict: Nhận diện viên thuốc (cần API Key)
"""

import os
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import monotonic
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Deque, Dict, Optional, Tuple
import torch
from dotenv import load_dotenv

# Đọc các biến môi trường từ file .env
load_dotenv()

# Import các module xử lý từ thư mục src
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
# Khởi tạo các model toàn cục
# ========================================
feature_extractor = None  # Trích xuất đặc điểm ảnh
classifier = None         # Phân loại viên thuốc
startup_state = {
    "ready": False,
    "error": None,
    "model_path": None,
}
INSECURE_API_KEYS = {
    "default-api-key-change-me",
    "pill-recognition-api-key-2024",
    "your-super-secret-api-key-change-me",
}

# Rate limit /predict theo client (in-memory)
RATE_LIMIT_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)
RATE_LIMIT_LOCK = Lock()
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
TRUST_X_FORWARDED_FOR = os.getenv("TRUST_X_FORWARDED_FOR", "false").lower() == "true"


def parse_cors_origins() -> list[str]:
    """Đọc danh sách CORS origins từ env, mặc định an toàn cho local dev."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    # Mặc định local cho development, production sẽ không mở CORS nếu không cấu hình
    if os.getenv("ENVIRONMENT", "development").lower() == "development":
        return ["http://localhost:8501", "http://127.0.0.1:8501"]

    return []


def resolve_client_id(request: Request) -> str:
    """Xác định định danh client để áp dụng rate-limit."""
    if TRUST_X_FORWARDED_FOR:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

    return request.client.host if request.client else "unknown-client"


def enforce_rate_limit(client_id: str, max_requests: int, window_seconds: int = 60) -> Tuple[int, int]:
    """Giới hạn số request theo sliding window in-memory."""
    if max_requests <= 0:
        return -1, 0

    now = monotonic()
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS[client_id]

        # Loại timestamp đã quá thời gian cửa sổ
        while bucket and (now - bucket[0]) > window_seconds:
            bucket.popleft()

        if len(bucket) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - bucket[0])) + 1) if bucket else window_seconds
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Quá nhiều request. Vui lòng thử lại sau.",
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(retry_after),
                },
            )

        bucket.append(now)
        remaining = max(0, max_requests - len(bucket))
        reset_after = max(1, int(window_seconds - (now - bucket[0])) + 1) if bucket else window_seconds
        return remaining, reset_after


# ========================================
# Định nghĩa các kiểu dữ liệu trả về
# ========================================

class PredictionResult(BaseModel):
    """Kết quả dự đoán viên thuốc."""
    success: bool                    # True nếu thành công, False nếu thất bại
    message: str                     # Thông báo kết quả
    predicted_pill: Optional[str] = None          # Mã viên thuốc dự đoán
    predicted_pill_display: Optional[str] = None  # Tên hiển thị của viên thuốc
    confidence: Optional[float] = None            # Độ tin cậy (0.0 - 1.0)
    probabilities: Optional[dict] = None          # Xác suất cho từng loại viên thuốc
    display_probabilities: Optional[dict] = None  # Xác suất với tên hiển thị
    error: Optional[str] = None                   # Thông báo lỗi nếu có


class HealthCheckResult(BaseModel):
    """Kết quả kiểm tra sức khỏe API."""
    status: str                      # "healthy" hoặc "unhealthy"
    models_loaded: bool              # True nếu các model đã được tải
    device: str                      # Thiết bị đang dùng (cpu hoặc cuda)
    startup_ready: bool              # True nếu startup thành công
    model_path: Optional[str] = None # Đường dẫn model
    startup_error: Optional[str] = None


# ========================================
# Quản lý vòng đời API (khởi động và tắt)
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hàm này chạy khi API khởi động và tắt."""
    global feature_extractor, classifier

    # Khi API khởi động
    try:
        startup_state["ready"] = False
        startup_state["error"] = None

        ghi_log("=== Khởi động Pill Recognition API ===")

        # Validate API key cấu hình
        api_key = os.getenv("API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Thiếu cấu hình API_KEY. Vui lòng thiết lập biến môi trường API_KEY.")
        if api_key in INSECURE_API_KEYS:
            raise RuntimeError("API_KEY đang dùng giá trị mặc định không an toàn. Hãy thay bằng khóa bí mật mới.")

        # Tải model trích xuất đặc điểm ảnh
        ghi_log("Đang tải Feature Extractor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_extractor = FeatureExtractor(ten_model="mobilenetv2", thiet_bi=device)
        ghi_log(f"Feature Extractor đã sẵn sàng | Thiết bị: {device}")

        # Tải model phân loại viên thuốc
        ghi_log("Đang tải ML Classifier...")
        classifier = PillClassifier(loai="svm")

        # Đọc model đã train từ file
        model_path = os.getenv("ML_CLASSIFIER_MODEL_PATH", "models/ml_classifier.pkl")
        startup_state["model_path"] = model_path

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Không tìm thấy model đã train tại: {model_path}")

        classifier.tai_model(model_path)
        if not classifier.da_train:
            raise RuntimeError("Model classifier chưa sẵn sàng sau khi tải")

        startup_state["ready"] = True

        ghi_log("=== API đã sẵn sàng ===")

    except Exception as e:
        startup_state["ready"] = False
        startup_state["error"] = str(e)
        ghi_loi(f"Không thể khởi động API: {str(e)}", e)
        raise

    yield

    # Khi API tắt
    startup_state["ready"] = False
    ghi_log("=== Đang tắt API ===")


# ========================================
# Khởi tạo FastAPI
# ========================================

app = FastAPI(
    title="Pill Recognition API",
    description="API nhận diện viên thuốc từ ảnh",
    version="1.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS (cho phép frontend gọi API)
cors_origins = parse_cors_origins()
allow_credentials = "*" not in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ========================================
# Các endpoint API
# ========================================

@app.get("/health", response_model=HealthCheckResult)
async def health_check():
    """Kiểm tra API có hoạt động bình thường không."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_loaded = (
            feature_extractor is not None
            and classifier is not None
            and getattr(classifier, "da_train", False)
        )

        payload = HealthCheckResult(
            status="healthy" if startup_state["ready"] and models_loaded else "unhealthy",
            models_loaded=models_loaded,
            device=device,
            startup_ready=bool(startup_state["ready"]),
            model_path=startup_state.get("model_path"),
            startup_error=startup_state.get("error"),
        )

        ghi_log("Health check được thực hiện")

        if payload.status == "unhealthy":
            return JSONResponse(status_code=503, content=payload.model_dump())

        return payload
    except Exception as e:
        ghi_loi(f"Health check thất bại: {str(e)}", e)
        raise HTTPException(status_code=500, detail="Health check thất bại")


@app.post("/predict", response_model=PredictionResult)
async def predict_pill(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    x_api_key: str = Header(None),
):
    """
    Nhận diện viên thuốc từ ảnh upload.
    
    Cách dùng:
    - Gửi POST request đến /predict
    - Đính kèm file ảnh (JPEG/PNG, tối đa 5MB)
    - Gửi header X-API-Key (API Key để xác thực)
    
    Trả về:
    - Kết quả dự đoán viên thuốc
    - Độ tin cậy
    - Xác suất cho từng loại
    """
    try:
        # Bước 0: Rate limit theo client IP
        client_id = resolve_client_id(request)
        remaining_requests, reset_after = enforce_rate_limit(client_id, RATE_LIMIT_REQUESTS_PER_MINUTE)
        if RATE_LIMIT_REQUESTS_PER_MINUTE > 0:
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(max(0, remaining_requests))
            response.headers["X-RateLimit-Reset"] = str(max(0, reset_after))

        # Bước 0.5: Kiểm tra trạng thái model
        if (
            not startup_state["ready"]
            or feature_extractor is None
            or classifier is None
            or not getattr(classifier, "da_train", False)
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model chưa sẵn sàng. Vui lòng thử lại sau.",
                headers={"Retry-After": "10"},
            )

        # Bước 1: Kiểm tra API Key
        try:
            kiem_tra_api_key(x_api_key)
            ghi_log("API Key hợp lệ")
        except HTTPException:
            ghi_bao_mat(f"Truy cập trái phép | API Key: {x_api_key[:10] if x_api_key else 'None'}...")
            raise

        # Bước 2: Kiểm tra file upload
        if not file.filename:
            ghi_bao_mat("Tên file rỗng")
            raise HTTPException(status_code=400, detail="Tên file là bắt buộc")

        ghi_log(f"Đang xử lý | File: {file.filename}")

        # Đọc nội dung file
        await file.seek(0)
        noi_dung = await file.read()

        # Nếu đọc không được, thử cách khác
        if not noi_dung:
            try:
                file.file.seek(0)
                noi_dung = file.file.read()
                ghi_log(f"Đọc fallback thành công | Kích thước: {len(noi_dung)} bytes")
            except Exception as e:
                ghi_loi(f"Đọc fallback thất bại: {str(e)}", e)

        # Kiểm tra file có rỗng không
        if not noi_dung or len(noi_dung) == 0:
            ghi_bao_mat("Nội dung file rỗng")
            raise HTTPException(status_code=400, detail="File upload rỗng")

        # Kiểm tra định dạng file (tên, kích thước, loại)
        kiem_tra_file_upload(file.filename, noi_dung)

        ghi_log(f"File hợp lệ | Kích thước: {len(noi_dung)} bytes")

        # Bước 3: Trích xuất đặc điểm ảnh bằng AI
        features = feature_extractor.trich_xuat_tu_anh(noi_dung)
        ghi_log(f"Đã trích xuất features | Shape: {features.shape}")

        # Bước 4: Phân loại viên thuốc bằng Machine Learning
        ket_qua = classifier.du_doan_mot_anh(features)

        # Bước 5: Định dạng kết quả để trả về
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lỗi nội bộ khi xử lý dự đoán",
        )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Trang chủ API - Hiển thị giao diện đẹp."""
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pill Recognition API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
                animation: fadeIn 0.8s ease-out;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
            }
            
            .logo {
                font-size: 80px;
                margin-bottom: 20px;
                animation: bounce 2s infinite;
            }
            
            @keyframes bounce {
                0%, 100% {
                    transform: translateY(0);
                }
                50% {
                    transform: translateY(-10px);
                }
            }
            
            h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .subtitle {
                color: #666;
                font-size: 1.2em;
                margin-bottom: 30px;
            }
            
            .version-badge {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 8px 20px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
                margin-bottom: 30px;
            }
            
            .description {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }
            
            .description h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .description p {
                color: #555;
                line-height: 1.6;
            }
            
            .endpoints {
                margin-bottom: 30px;
            }
            
            .endpoints h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5em;
            }
            
            .endpoint-card {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 15px;
                transition: all 0.3s ease;
                border: 2px solid transparent;
            }
            
            .endpoint-card:hover {
                transform: translateX(10px);
                border-color: #667eea;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
            }
            
            .endpoint-method {
                display: inline-block;
                padding: 5px 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 0.9em;
                margin-right: 10px;
            }
            
            .method-get {
                background: #28a745;
                color: white;
            }
            
            .method-post {
                background: #007bff;
                color: white;
            }
            
            .endpoint-path {
                font-family: 'Courier New', monospace;
                font-weight: bold;
                color: #333;
                font-size: 1.1em;
            }
            
            .endpoint-desc {
                color: #666;
                margin-top: 10px;
                padding-left: 75px;
            }
            
            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .feature-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                transition: all 0.3s ease;
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }
            
            .feature-icon {
                font-size: 40px;
                margin-bottom: 15px;
            }
            
            .feature-title {
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .feature-desc {
                font-size: 0.9em;
                opacity: 0.9;
            }
            
            .cta-section {
                text-align: center;
                margin-top: 40px;
            }
            
            .cta-button {
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border-radius: 30px;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.1em;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            
            .cta-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
            }
            
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #999;
                font-size: 0.9em;
            }
            
            .tech-stack {
                display: flex;
                justify-content: center;
                gap: 15px;
                margin-top: 15px;
                flex-wrap: wrap;
            }
            
            .tech-badge {
                background: #f0f0f0;
                padding: 5px 15px;
                border-radius: 15px;
                font-size: 0.85em;
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">💊</div>
                <h1>Pill Recognition API</h1>
                <p class="subtitle">Hệ thống nhận diện viên thuốc thông minh</p>
                <span class="version-badge">v1.0.0</span>
            </div>
            
            <div class="description">
                <h3>🎯 Giới thiệu</h3>
                <p>
                    API nhận diện viên thuốc sử dụng công nghệ Hybrid Deep Learning + Machine Learning.
                    Hệ thống có thể phân tích ảnh viên thuốc và trả về kết quả nhận diện với độ chính xác cao.
                </p>
            </div>
            
            <div class="features">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Nhận diện chính xác</div>
                    <div class="feature-desc">Sử dụng AI để phân tích ảnh</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <div class="feature-title">Xử lý nhanh</div>
                    <div class="feature-desc">Kết quả trong vài giây</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔒</div>
                    <div class="feature-title">Bảo mật</div>
                    <div class="feature-desc">Xác thực bằng API Key</div>
                </div>
            </div>
            
            <div class="endpoints">
                <h2>📚 API Endpoints</h2>
                
                <div class="endpoint-card">
                    <span class="endpoint-method method-get">GET</span>
                    <span class="endpoint-path">/health</span>
                    <div class="endpoint-desc">
                        Kiểm tra trạng thái hoạt động của API và các model AI
                    </div>
                </div>
                
                <div class="endpoint-card">
                    <span class="endpoint-method method-post">POST</span>
                    <span class="endpoint-path">/predict</span>
                    <div class="endpoint-desc">
                        Nhận diện viên thuốc từ ảnh upload (cần X-API-Key header)
                    </div>
                </div>
                
                <div class="endpoint-card">
                    <span class="endpoint-method method-get">GET</span>
                    <span class="endpoint-path">/docs</span>
                    <div class="endpoint-desc">
                        Xem tài liệu API tương tác (Swagger UI)
                    </div>
                </div>
                
                <div class="endpoint-card">
                    <span class="endpoint-method method-get">GET</span>
                    <span class="endpoint-path">/redoc</span>
                    <div class="endpoint-desc">
                        Xem tài liệu API (ReDoc format)
                    </div>
                </div>
            </div>
            
            <div class="cta-section">
                <a href="/docs" class="cta-button">📖 Xem tài liệu API</a>
            </div>
            
            <div class="footer">
                <p>Pill Recognition System © 2026</p>
                <div class="tech-stack">
                    <span class="tech-badge">FastAPI</span>
                    <span class="tech-badge">PyTorch</span>
                    <span class="tech-badge">scikit-learn</span>
                    <span class="tech-badge">MobileNetV2</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ========================================
# Xử lý lỗi
# ========================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Xử lý các lỗi HTTP theo mức độ phù hợp."""
    if exc.status_code >= 500:
        ghi_loi(f"[HTTP {exc.status_code}] {request.url.path} | {exc.detail}")
    elif exc.status_code in {401, 403, 429}:
        ghi_bao_mat(f"[HTTP {exc.status_code}] {request.url.path} | {exc.detail}")
    else:
        ghi_log(f"[HTTP {exc.status_code}] {request.url.path} | {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": "Lỗi",
            "error": exc.detail,
        },
        headers=exc.headers or {},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Bắt lỗi chưa xử lý để trả về HTTP 500 chuẩn."""
    ghi_loi(f"Unhandled exception tại {request.url.path}: {str(exc)}", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "Lỗi",
            "error": "Lỗi nội bộ server",
        },
    )


# ========================================
# Chạy server
# ========================================

if __name__ == "__main__":
    import uvicorn

    # Đọc cấu hình từ biến môi trường
    host = os.getenv("BACKEND_HOST", "127.0.0.1")
    port = int(os.getenv("BACKEND_PORT", 8000))

    ghi_log(f"Khởi động server trên {host}:{port}")

    # Chạy server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=None,
    )