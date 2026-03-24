"""
File này quản lý mapping giữa class label (số) và tên thuốc (text).

Mục đích:
    - Model chỉ trả về class label (số: 0, 1, 2, ...)
    - Module này chuyển số đó thành tên thuốc dễ hiểu
    - VD: Class 0 → "Bivinadadol 500mg"

Cách sử dụng:
    from src.pill_label_utils import lay_ten_thuoc, load_mapping
    
    mapping = load_mapping()
    ten_thuoc = lay_ten_thuoc("0", mapping)  # → "Bivinadadol 500mg"
"""

import json
from pathlib import Path

from .path_utils import tim_file_trong_myenv


# ========================================
# Cấu hình
# ========================================

# File chứa mapping class label → tên thuốc
FILE_MAPPING = tim_file_trong_myenv("pill_labels.json")

# Phạm vi class labels (0-252)
BAT_DAU = 0
KET_THUC = 252

# Mapping mặc định (hardcoded)
# Nếu không có trong file JSON, sẽ dùng mapping này
MAPPING_MAC_DINH = {
    "0": "Bivinadadol 500mg",
    "1": "Actadol 500mg",
    "2": "Paracetamol 500mg",
    "3": "Amoxicillin 500mg",
    "4": "Ibuprofen 400mg",
    "5": "Cefuroxime 250mg",
    "6": "Loratadine 10mg",
    "7": "Omeprazole 20mg",
    "8": "Diclofenac 50mg",
    "9": "Metformin 850mg",
    "10": "Allerphast 180mg",
    # ... (còn 242 thuốc khác)
}


# ========================================
# Các hàm chính
# ========================================

def tao_mapping_day_du() -> dict:
    """
    Tạo mapping đầy đủ cho tất cả class labels.
    Nếu chưa có tên thuốc, dùng placeholder "Class X".
    
    Returns:
        Dict mapping {class_label: pill_name}
    """
    mapping = {str(i): f"Class {i}" for i in range(BAT_DAU, KET_THUC + 1)}
    mapping.update(MAPPING_MAC_DINH)
    return mapping


def chuan_hoa_label(label) -> str:
    """
    Chuẩn hóa class label về string.
    
    Args:
        label: Class label (có thể là int, str, ...)
    
    Returns:
        Class label dạng string
    """
    return str(label).strip()


def load_mapping() -> dict:
    """
    Tải mapping từ file JSON. Nếu file không tồn tại, dùng mapping mặc định.
    
    Returns:
        Dict mapping {class_label: pill_name}
    """
    mapping = tao_mapping_day_du()

    try:
        if FILE_MAPPING.exists():
            raw = json.loads(FILE_MAPPING.read_text(encoding="utf-8"))
            mapping.update({
                chuan_hoa_label(k): str(v).strip()
                for k, v in raw.items()
                if str(k).strip() and str(v).strip()
            })
    except Exception:
        pass

    return mapping


def luu_mapping(mapping: dict) -> Path:
    """
    Lưu mapping ra file JSON.
    
    Args:
        mapping: Dict mapping {class_label: pill_name}
    
    Returns:
        Đường dẫn file đã lưu
    """
    # Kết hợp với mapping mặc định
    mapping_chuan = tao_mapping_day_du()
    mapping_chuan.update({
        chuan_hoa_label(k): str(v).strip()
        for k, v in mapping.items()
        if str(k).strip() and str(v).strip()
    })

    # Tạo thư mục nếu chưa có
    FILE_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    
    # Ghi file
    FILE_MAPPING.write_text(
        json.dumps(mapping_chuan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FILE_MAPPING


def lay_ten_thuoc(label, mapping: dict = None) -> str:
    """
    Lấy tên thuốc từ class label.
    
    Args:
        label: Class label (số hoặc string)
        mapping: Mapping tùy chọn (nếu None, tự động load)
    
    Returns:
        Tên thuốc dễ đọc
    
    Ví dụ:
        lay_ten_thuoc("0")  # → "Bivinadadol 500mg"
        lay_ten_thuoc("999")  # → "Class 999"
    """
    if mapping is None:
        mapping = load_mapping()
    
    label_chuan = chuan_hoa_label(label)
    ten_thuoc = mapping.get(label_chuan)

    # Nếu là placeholder hoặc rỗng, trả về "Class X"
    if not ten_thuoc or ten_thuoc in {"", f"Class {label_chuan}", f"TODO {label_chuan}"}:
        return f"Class {label_chuan}"

    return ten_thuoc


def kiem_tra_co_ten_thuoc(label, mapping: dict = None) -> bool:
    """
    Kiểm tra class đã có tên thuốc thật hay chưa.
    
    Args:
        label: Class label
        mapping: Mapping tùy chọn
    
    Returns:
        True nếu đã có tên thật, False nếu vẫn là placeholder
    """
    if mapping is None:
        mapping = load_mapping()
    
    label_chuan = chuan_hoa_label(label)
    ten_thuoc = mapping.get(label_chuan)

    if not ten_thuoc:
        return False

    # Kiểm tra có phải placeholder không
    return ten_thuoc not in {"", f"Class {label_chuan}", f"TODO {label_chuan}"}


def format_hien_thi(label, mapping: dict = None) -> str:
    """
    Format tên thuốc để hiển thị trên UI/API.
    
    Args:
        label: Class label
        mapping: Mapping tùy chọn
    
    Returns:
        Chuỗi hiển thị (VD: "Paracetamol 500mg (ID: 2)")
    """
    if mapping is None:
        mapping = load_mapping()
    
    label_chuan = chuan_hoa_label(label)
    ten_thuoc = mapping.get(label_chuan)

    if ten_thuoc and ten_thuoc not in {"", f"Class {label_chuan}", f"TODO {label_chuan}"}:
        return f"{ten_thuoc} (ID: {label_chuan})"

    return f"Class {label_chuan}"


def chuyen_probabilities_sang_ten(probabilities: dict, mapping: dict = None) -> dict:
    """
    Chuyển key của probabilities thành tên thuốc dễ đọc.
    
    Args:
        probabilities: Dict {class_label: probability}
        mapping: Mapping tùy chọn
    
    Returns:
        Dict {pill_name: probability}
    """
    if mapping is None:
        mapping = load_mapping()
    
    return {
        format_hien_thi(label, mapping): prob
        for label, prob in probabilities.items()
    }