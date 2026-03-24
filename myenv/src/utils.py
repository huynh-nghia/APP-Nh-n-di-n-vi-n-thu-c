"""
File này xử lý ảnh - Chuyển ảnh thành dữ liệu cho AI.

Mục đích:
    - Đọc ảnh từ file hoặc bytes
    - Resize ảnh về kích thước chuẩn (224x224)
    - Chuyển ảnh thành tensor để đưa vào model
    - Normalize pixel values

Tại sao cần normalize?
    - Model được train với giá trị pixel chuẩn hóa
    - Nếu không normalize, model sẽ cho kết quả sai
"""

import io
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms

from .logger import ghi_log, ghi_loi


# ========================================
# Cấu hình mặc định
# ========================================

# Kích thước ảnh đầu vào cho model (224x224 pixels)
KICH_THUOC_ANH = 224

# Giá trị trung bình và độ lệch chuẩn của ImageNet
# (Model được train trên ImageNet nên phải dùng đúng giá trị này)
MEAN = [0.485, 0.456, 0.406]  # RGB
STD = [0.229, 0.224, 0.225]   # RGB


# ========================================
# Các hàm đọc ảnh
# ========================================

def doc_anh_tu_bytes(du_lieu: bytes) -> Image.Image:
    """
    Đọc ảnh từ dữ liệu bytes (khi người dùng upload file).
    
    Args:
        du_lieu: Nội dung file ảnh dạng bytes
    
    Returns:
        PIL Image object
    
    Ví dụ:
        anh = doc_anh_tu_bytes(file_bytes)
    """
    try:
        anh = Image.open(io.BytesIO(du_lieu))
        ghi_log(f"Đã đọc ảnh từ bytes. Kích thước: {anh.size}")
        return anh
    except Exception as e:
        ghi_loi(f"Không thể đọc ảnh từ bytes: {str(e)}", e)
        raise ValueError(f"File ảnh không hợp lệ: {str(e)}")


def doc_anh_tu_file(duong_dan: str) -> Image.Image:
    """
    Đọc ảnh từ đường dẫn file.
    
    Args:
        duong_dan: Đường dẫn đến file ảnh
    
    Returns:
        PIL Image object
    
    Ví dụ:
        anh = doc_anh_tu_file("/path/to/image.jpg")
    """
    try:
        # Kiểm tra file có tồn tại không
        if not Path(duong_dan).exists():
            raise FileNotFoundError(f"Không tìm thấy file ảnh: {duong_dan}")

        anh = Image.open(duong_dan)
        ghi_log(f"Đã đọc ảnh từ: {duong_dan}. Kích thước: {anh.size}")
        return anh
    except Exception as e:
        ghi_loi(f"Không thể đọc ảnh từ file: {str(e)}", e)
        raise


# ========================================
# Các hàm xử lý ảnh
# ========================================

def resize_anh(anh: Image.Image, kich_thuoc: int = KICH_THUOC_ANH) -> Image.Image:
    """
    Resize ảnh về kích thước vuông (kich_thuoc x kich_thuoc).
    
    Args:
        anh: PIL Image object
        kich_thuoc: Kích thước mục tiêu (pixels)
    
    Returns:
        Ảnh đã resize (chuyển sang RGB nếu cần)
    """
    try:
        # Chuyển sang RGB nếu ảnh có alpha channel (PNG trong suốt)
        if anh.mode != "RGB":
            anh = anh.convert("RGB")

        # Resize sử dụng LANCZOS (giữ chất lượng tốt)
        anh_da_resize = anh.resize((kich_thuoc, kich_thuoc), Image.Resampling.LANCZOS)
        ghi_log(f"Đã resize ảnh thành {kich_thuoc}x{kich_thuoc}")
        return anh_da_resize
    except Exception as e:
        ghi_loi(f"Không thể resize ảnh: {str(e)}", e)
        raise


def anh_sang_tensor(anh: Image.Image, kich_thuoc: int = KICH_THUOC_ANH) -> torch.Tensor:
    """
    Chuyển đổi ảnh PIL thành tensor PyTorch (đã normalize).
    
    Quy trình:
        1. Resize ảnh về kích thước chuẩn
        2. Chuyển thành tensor (giá trị 0-1)
        3. Normalize theo mean/std của ImageNet
        4. Thêm batch dimension (1, 3, 224, 224)
    
    Args:
        anh: PIL Image object
        kich_thuoc: Kích thước target (pixels)
    
    Returns:
        Tensor shape (1, 3, kich_thuoc, kich_thuoc)
    """
    try:
        # Resize ảnh
        anh = resize_anh(anh, kich_thuoc)

        # Pipeline chuyển đổi:
        # 1. ToTensor: PIL Image → Tensor (0-1)
        # 2. Normalize: Chuẩn hóa theo ImageNet
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        # Chuyển ảnh thành tensor và thêm batch dimension
        tensor = transform(anh).unsqueeze(0)  # Shape: (1, 3, kich_thuoc, kich_thuoc)
        ghi_log(f"Đã chuyển ảnh thành tensor. Shape: {tensor.shape}")
        return tensor
    except Exception as e:
        ghi_loi(f"Không thể chuyển ảnh thành tensor: {str(e)}", e)
        raise


# ========================================
# Các hàm chuyển đổi dữ liệu
# ========================================

def numpy_sang_tensor(mang: np.ndarray) -> torch.Tensor:
    """
    Chuyển NumPy array thành PyTorch tensor.
    
    Args:
        mang: NumPy array
    
    Returns:
        PyTorch tensor (float)
    """
    return torch.from_numpy(mang).float()


def tensor_sang_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Chuyển PyTorch tensor thành NumPy array.
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        NumPy array
    """
    return tensor.detach().cpu().numpy()


def luu_anh(anh: Image.Image, duong_dan: str):
    """
    Lưu ảnh PIL vào file.
    
    Args:
        anh: PIL Image object
        duong_dan: Đường dẫn file output
    """
    try:
        Path(duong_dan).parent.mkdir(parents=True, exist_ok=True)
        anh.save(duong_dan)
        ghi_log(f"Đã lưu ảnh vào: {duong_dan}")
    except Exception as e:
        ghi_loi(f"Không thể lưu ảnh: {str(e)}", e)
        raise