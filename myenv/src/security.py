"""
File này xử lý bảo mật cho API.

Mục đích:
    - Xác thực API Key
    - Kiểm tra file upload (định dạng, kích thước, nội dung)
    - Chống tấn công (directory traversal, file độc hại)

Tại sao cần bảo mật?
    - API công khai có thể bị spam
    - File upload có thể chứa virus
    - Cần giới hạn kích thước file
"""

import os
import magic
from pathlib import Path
from fastapi import HTTPException, status

from .logger import ghi_bao_mat, ghi_loi


# ========================================
# Cấu hình
# ========================================

# Kích thước file tối đa (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Các định dạng ảnh được phép
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Các MIME type được phép
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# ========================================
# Các hàm kiểm tra
# ========================================

def kiem_tra_api_key(api_key: str, expected_key: str = None) -> bool:
    """
    Xác thực API Key từ header request.
    
    Args:
        api_key: API Key từ header X-API-Key
        expected_key: API Key mong đợi (nếu None, lấy từ environment)
    
    Returns:
        True nếu hợp lệ
    
    Raises:
        HTTPException: Nếu API Key không hợp lệ (401)
    """
    if expected_key is None:
        expected_key = os.getenv("API_KEY", "default-api-key-change-me")

    if not api_key or api_key != expected_key:
        ghi_bao_mat(f"API Key không hợp lệ: {api_key[:10] if api_key else 'None'}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key không hợp lệ",
        )

    return True


def kiem_tra_ten_file(ten_file: str) -> bool:
    """
    Kiểm tra tên file để chống directory traversal.
    
    Args:
        ten_file: Tên file cần kiểm tra
    
    Returns:
        True nếu an toàn
    
    Raises:
        HTTPException: Nếu tên file không an toàn (400)
    """
    if ".." in ten_file or "/" in ten_file or "\\" in ten_file:
        ghi_bao_mat(f"Phát hiện tên file độc hại: {ten_file}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên file không hợp lệ",
        )

    return True


def kiem_tra_kich_thuoc(kich_thuoc: int, max_size: int = MAX_FILE_SIZE) -> bool:
    """
    Kiểm tra kích thước file.
    
    Args:
        kich_thuoc: Kích thước file (bytes)
        max_size: Kích thước tối đa (bytes)
    
    Returns:
        True nếu hợp lệ
    
    Raises:
        HTTPException: Nếu file quá lớn (413)
    """
    if kich_thuoc > max_size:
        ghi_bao_mat(f"File quá lớn: {kich_thuoc} bytes > {max_size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File quá lớn. Tối đa {max_size / (1024 * 1024):.1f}MB",
        )

    return True


def kiem_tra_dinh_dang(ten_file: str, allowed: set = ALLOWED_EXTENSIONS) -> bool:
    """
    Kiểm tra phần mở rộng của file.
    
    Args:
        ten_file: Tên file
        allowed: Tập hợp các phần mở rộng được phép
    
    Returns:
        True nếu hợp lệ
    
    Raises:
        HTTPException: Nếu định dạng không hợp lệ (400)
    """
    ext = Path(ten_file).suffix.lower()

    if ext not in allowed:
        ghi_bao_mat(f"Định dạng file không hợp lệ: {ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chỉ chấp nhận file: {', '.join(allowed)}",
        )

    return True


def kiem_tra_mime_type(noi_dung: bytes, allowed: set = ALLOWED_MIME_TYPES) -> bool:
    """
    Kiểm tra MIME type thực tế của file.
    
    Args:
        noi_dung: Nội dung file dạng bytes
        allowed: Tập hợp MIME types được phép
    
    Returns:
        True nếu hợp lệ
    
    Raises:
        HTTPException: Nếu MIME type không hợp lệ (400)
    """
    try:
        # Kiểm tra file có rỗng không
        if not noi_dung or len(noi_dung) == 0:
            ghi_bao_mat("Nội dung file rỗng")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File rỗng. Vui lòng upload file ảnh hợp lệ.",
            )
        
        # Kiểm tra MIME type
        mime = magic.from_buffer(noi_dung, mime=True)

        if mime not in allowed:
            ghi_bao_mat(f"MIME type không hợp lệ: {mime}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File không phải ảnh hợp lệ (JPEG/PNG)",
            )

        return True
    except HTTPException:
        raise
    except Exception as e:
        ghi_loi(f"Lỗi khi kiểm tra MIME type: {str(e)}", e)
        # Fallback: Bỏ qua kiểm tra MIME nếu magic library lỗi
        return True


def kiem_tra_file_upload(ten_file: str, noi_dung: bytes) -> bool:
    """
    Tổng hợp kiểm tra file upload.
    
    Args:
        ten_file: Tên file
        noi_dung: Nội dung file dạng bytes
    
    Returns:
        True nếu file hợp lệ
    
    Raises:
        HTTPException: Nếu file không hợp lệ
    """
    kiem_tra_ten_file(ten_file)
    kiem_tra_dinh_dang(ten_file)
    kiem_tra_kich_thuoc(len(noi_dung))
    kiem_tra_mime_type(noi_dung)

    return True