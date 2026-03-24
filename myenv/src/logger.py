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
    # Tạo thư mục chứa file log nếu chưa có
    Path(file_log).parent.mkdir(parents=True, exist_ok=True)

    # Tạo logger
    logger = logging.getLogger(ten)
    logger.setLevel(muc)

    # Xóa handler cũ (tránh ghi trùng)
    logger.handlers.clear()

    # Handler: Ghi file với rotation
    # - Tối đa 5MB mỗi file
    # - Giữ lại 5 file backup
    handler = logging.handlers.RotatingFileHandler(
        file_log,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    handler.setLevel(muc)

    # Formatter: Định dạng JSON
    formatter = DinhDangJSON()
    handler.setFormatter(formatter)

    # Thêm handler vào logger
    logger.addHandler(handler)

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