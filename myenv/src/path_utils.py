"""
File này giúp tìm đúng đường dẫn đến file,不管你 chạy code từ thư mục nào.

Ví dụ:
    - Bạn chạy code từ thư mục /home/user/
    - Nhưng file ảnh nằm trong /home/user/Documents/pill_recognition_app/data/
    - File này sẽ tự động tìm đúng chỗ
"""

from pathlib import Path

# ========================================
# Xác định các thư mục chính
# ========================================

# Thư mục chứa file này: myenv/src/
THU_MUC_HIEN_TAI = Path(__file__).resolve().parent

# Thư mục myenv/ (cha của src/)
THU_MUC_MYENV = THU_MUC_HIEN_TAI.parent

# Thư mục gốc dự án (cha của myenv/)
THU_MUC_GOC = THU_MUC_MYENV.parent


def tim_file_data(ten_file: str) -> Path:
    """
    Tìm file trong thư mục data.
    
    Cách tìm:
        1. Nếu ten_file là đường dẫn đầy đủ (VD: /home/user/data/abc.jpg) → Dùng luôn
        2. Tìm trong thư mục hiện tại
        3. Tìm trong myenv/
        4. Tìm trong thư mục gốc dự án
    
    Args:
        ten_file: Tên file cần tìm (VD: "data", "models/ml_classifier.pkl")
    
    Returns:
        Đường dẫn đến file
    
    Ví dụ:
        tim_file_data("data")  # → /home/user/Documents/pill_recognition_app/data
        tim_file_data("models/ml_classifier.pkl")  # → /home/user/Documents/pill_recognition_app/myenv/models/ml_classifier.pkl
    """
    # Tạo Path object
    duong_dan = Path(ten_file)
    
    # Nếu là đường dẫn tuyệt đối → trả về luôn
    if duong_dan.is_absolute():
        return duong_dan
    
    # Danh sách các chỗ có thể chứa file
    cac_cho_tim = [
        Path.cwd() / ten_file,           # Thư mục hiện tại
        THU_MUC_MYENV / ten_file,        # myenv/
        THU_MUC_GOC / ten_file,          # Thư mục gốc dự án
    ]
    
    # Tìm chỗ đầu tiên có file
    for cho in cac_cho_tim:
        if cho.exists():
            return cho
    
    # Nếu không tìm thấy, trả về vị trí mặc định
    return THU_MUC_GOC / ten_file


def tim_file_trong_myenv(ten_file: str) -> Path:
    """
    Tìm file trong thư mục myenv/.
    
    Dùng cho các file nội bộ như models, logs, pill_labels.json
    
    Args:
        ten_file: Tên file (VD: "models/ml_classifier.pkl", "logs/app.log")
    
    Returns:
        Đường dẫn đến file trong myenv/
    
    Ví dụ:
        tim_file_trong_myenv("models/ml_classifier.pkl")
        # → /home/user/Documents/pill_recognition_app/myenv/models/ml_classifier.pkl
    """
    duong_dan = Path(ten_file)
    
    # Nếu là đường dẫn tuyệt đối → trả về luôn
    if duong_dan.is_absolute():
        return duong_dan
    
    # Trả về đường dẫn trong myenv/
    return THU_MUC_MYENV / ten_file