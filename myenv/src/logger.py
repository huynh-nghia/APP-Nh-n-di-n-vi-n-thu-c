"""
File này ghi lại mọi hoạt động của ứng dụng để dễ debug.

Mục đích:
    - Ghi log khi ứng dụng chạy
    - Ghi log khi có lỗi
    - Ghi log khi có sự kiện bảo mật

Cách sử dụng:
    from src.logger import ghi_log, ghi_loi, ghi_bao_mat
    
    ghi_log("Ứng dụng đã khởi động")
    ghi_loi("Có lỗi xảy ra", exception=e)
    ghi_bao_mat("Phát hiện truy cập trái phép")
"""

import logging
import logging.handlers
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


class DinhDangJSON(logging.Formatter):
    """
    Chuyển đổi log thành định dạng JSON.
    
    Ví dụ output:
        {"timestamp": "2024-01-15T10:30:00", "level": "INFO", "message": "App started"}
    """

    def format(self, record):
        """Chuyển record log thành JSON string."""
        
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),  # Thời gian
            "level": record.levelname,                    # Mức độ (INFO, ERROR, ...)
            "message": record.getMessage(),               # Nội dung
            "module": record.module,                      # Tên module
            "function": record.funcName,                  # Tên hàm
            "line": record.lineno,                        # Số dòng
        }

        # Nếu có exception, thêm thông tin
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def lay_duong_dan_log_mac_dinh() -> Path:
    """Trả về thư mục log ưu tiên từ env hoặc thư mục logs trong myenv/."""
    log_dir = os.getenv("LOG_DIR", "").strip()
    if log_dir:
        return Path(log_dir)

    return Path(__file__).resolve().parent.parent / "logs"


def dam_bao_thu_muc_log_co_the_ghi(thu_muc_log: Path) -> Path:
    """Đảm bảo có 1 thư mục log writable; fallback sang /tmp nếu cần."""
    cac_thu_muc_ung_vien = [
        thu_muc_log,
        Path(tempfile.gettempdir()) / "pill-recognition-logs",
    ]

    for thu_muc in cac_thu_muc_ung_vien:
        try:
            thu_muc.mkdir(parents=True, exist_ok=True)
            file_thu = thu_muc / ".write_test"
            file_thu.write_text("ok", encoding="utf-8")
            file_thu.unlink(missing_ok=True)
            return thu_muc
        except OSError:
            continue

    raise PermissionError(
        f"Không có thư mục log nào có thể ghi được. Đã thử: {', '.join(str(p) for p in cac_thu_muc_ung_vien)}"
    )


def tao_logger(ten: str, file_log: str, muc=logging.INFO):
    """
    Tạo một logger mới.
    
    Args:
        ten: Tên logger (VD: "app", "error", "security")
        file_log: Đường dẫn file log (VD: "logs/app.log")
        muc: Mức độ log (INFO, ERROR, WARNING)
    
    Returns:
        Logger đã cấu hình
    """
    # Xác định thư mục log có thể ghi được
    ten_file_log = Path(file_log).name
    thu_muc_log = dam_bao_thu_muc_log_co_the_ghi(lay_duong_dan_log_mac_dinh())
    duong_dan_log = thu_muc_log / ten_file_log

    # Tạo logger
    logger = logging.getLogger(ten)
    logger.setLevel(muc)
    logger.propagate = False

    # Xóa handler cũ (tránh ghi trùng)
    logger.handlers.clear()

    # Handler: Ghi file với rotation
    # - Tối đa 5MB mỗi file
    # - Giữ lại 5 file backup
    # Formatter: Định dạng JSON
    formatter = DinhDangJSON()

    handler = logging.handlers.RotatingFileHandler(
        duong_dan_log,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(muc)
    handler.setFormatter(formatter)

    # Thêm handler vào logger
    logger.addHandler(handler)

    # Đồng thời log ra stdout để dễ xem trên Docker/Hugging Face
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(muc)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ========================================
# Tạo các logger chính
# ========================================

# Logger cho hoạt động thông thường
logger_app = tao_logger("app", "logs/app.log", logging.INFO)

# Logger cho lỗi
logger_loi = tao_logger("error", "logs/error.log", logging.ERROR)

# Logger cho bảo mật
logger_bao_mat = tao_logger("security", "logs/security.log", logging.WARNING)


# ========================================
# Các hàm ghi log đơn giản
# ========================================

def ghi_log(loi_nhan: str):
    """
    Ghi log hoạt động thông thường.
    
    Args:
        loi_nhan: Nội dung cần ghi
    
    Ví dụ:
        ghi_log("Người dùng đã upload ảnh")
        ghi_log("Đang xử lý...")
    """
    logger_app.info(loi_nhan)


def ghi_loi(loi_nhan: str, exception=None):
    """
    Ghi log lỗi.
    
    Args:
        loi_nhan: Mô tả lỗi
        exception: Exception object (nếu có)
    
    Ví dụ:
        ghi_loi("Không thể load model")
        ghi_loi("Lỗi khi predict", exception=e)
    """
    if exception:
        logger_loi.exception(f"{loi_nhan} | Exception: {str(exception)}")
    else:
        logger_loi.error(loi_nhan)


def ghi_bao_mat(loi_nhan: str):
    """
    Ghi log sự kiện bảo mật.
    
    Args:
        loi_nhan: Mô tả sự kiện
    
    Ví dụ:
        ghi_bao_mat("API Key không hợp lệ")
        ghi_bao_mat("Phát hiện file độc hại")
    """
    logger_bao_mat.warning(f"[SECURITY] {loi_nhan}")