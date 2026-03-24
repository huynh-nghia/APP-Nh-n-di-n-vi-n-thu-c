"""
File này huấn luyện mô hình phân loại viên thuốc.

Mục đích:
    - Đọc ảnh từ thư mục data/
    - Trích xuất features bằng Feature Extractor (DL)
    - Train Classifier (ML)
    - Lưu model đã train vào models/

Quy trình:
    1. Load dataset (ảnh + nhãn)
    2. Trích xuất feature vector cho mỗi ảnh
    3. Train classifier trên feature vectors
    4. Lưu model ra file
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple

from .feature_extractor import FeatureExtractor
from .classifier import PillClassifier
from .logger import ghi_log, ghi_loi
from .path_utils import tim_file_data, tim_file_trong_myenv


def lay_class_tu_ten_file(ten_file: str) -> Tuple[int, str]:
    """
    Lấy class label và góc chụp từ tên file.
    
    Định dạng: (class_label)view_type.ext
    Ví dụ: 
        - (0)r15.jpg → class=0, view=r15
        - (10)outline.jpg → class=10, view=outline
    
    Args:
        ten_file: Tên file ảnh
    
    Returns:
        Tuple (class_label, view_type)
    """
    pattern = r'\((\d+)\)(r15|r30|outline)\.(jpg|jpeg|png)$'
    match = re.match(pattern, ten_file, re.IGNORECASE)

    if not match:
        raise ValueError(f"Tên file không đúng định dạng: {ten_file}")

    return int(match.group(1)), match.group(2).lower()


def load_dataset(duong_dan: str, split: str = "train") -> Tuple[List[str], List[int]]:
    """
    Tải danh sách ảnh và nhãn từ thư mục data.
    
    Args:
        duong_dan: Đường dẫn đến thư mục data
        split: Loại split ("train", "test", hoặc "val")
    
    Returns:
        Tuple (image_paths, labels)
    """
    duong_dan_data = tim_file_data(duong_dan)

    if not duong_dan_data.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục data: {duong_dan}")

    # Xác định thư mục split
    if split in ["train", "test", "val"]:
        thu_muc = duong_dan_data / split
        if not thu_muc.exists():
            # Nếu không có thư mục split, dùng thư mục gốc
            thu_muc = duong_dan_data
    else:
        thu_muc = duong_dan_data

    danh_sach_anh = []
    danh_sach_nhan = []

    # Duyệt qua tất cả file
    for file_path in sorted(thu_muc.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                class_label, _ = lay_class_tu_ten_file(file_path.name)
                danh_sach_anh.append(str(file_path))
                danh_sach_nhan.append(class_label)
                ghi_log(f"Đã tìm thấy: {file_path.name} | Class: {class_label}")
            except ValueError:
                ghi_log(f"Bỏ qua file: {file_path.name}")

    if len(danh_sach_anh) == 0:
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong {thu_muc}")

    ghi_log(f"Đã load dataset | Split: {split} | Tổng ảnh: {len(danh_sach_anh)} | Số lớp: {len(set(danh_sach_nhan))}")

    return danh_sach_anh, danh_sach_nhan


def trich_xuat_features(danh_sach_anh: List[str], feature_extractor: FeatureExtractor) -> np.ndarray:
    """
    Trích xuất features từ tất cả ảnh.
    
    Args:
        danh_sach_anh: Danh sách đường dẫn ảnh
        feature_extractor: Feature extractor model
    
    Returns:
        Feature matrix shape (n_samples, feature_dim)
    """
    danh_sach_features = []

    ghi_log(f"Bắt đầu trích xuất features từ {len(danh_sach_anh)} ảnh...")

    for i, duong_dan in enumerate(danh_sach_anh):
        try:
            features = feature_extractor.trich_xuat_tu_anh(duong_dan)
            danh_sach_features.append(features)

            if (i + 1) % 10 == 0:
                ghi_log(f"Tiến độ: {i + 1}/{len(danh_sach_anh)} ảnh")

        except Exception as e:
            ghi_loi(f"Không thể trích xuất features từ {duong_dan}: {str(e)}", e)
            continue

    if len(danh_sach_features) == 0:
        raise ValueError("Không trích xuất được features")

    return np.array(danh_sach_features)


def train_model(
    duong_dan_data: str = "data",
    duong_dan_model: str = "models/ml_classifier.pkl",
    loai_classifier: str = "svm",
    feature_extractor_model: str = "mobilenetv2",
    thiet_bi: str = None,
    so_vong: int = None,
) -> dict:
    """
    Train mô hình phân loại viên thuốc.
    
    Args:
        duong_dan_data: Đường dẫn thư mục data
        duong_dan_model: Đường dẫn lưu model
        loai_classifier: Loại classifier ("svm" hoặc "random_forest")
        feature_extractor_model: Loại feature extractor ("mobilenetv2" hoặc "resnet18")
        thiet_bi: Device ("cuda" hoặc "cpu")
        so_vong: Số vòng lặp train
    
    Returns:
        Dict chứa thông tin training
    """
    try:
        if so_vong is not None and so_vong < 1:
            raise ValueError("so_vong phải >= 1")

        ghi_log("=" * 60)
        ghi_log("Bắt đầu huấn luyện mô hình")
        ghi_log("=" * 60)

        # Bước 1: Load dataset
        ghi_log(f"Load dataset từ: {duong_dan_data}")
        danh_sach_anh, danh_sach_nhan = load_dataset(duong_dan_data, split="train")

        # Bước 2: Khởi tạo Feature Extractor
        ghi_log(f"Khởi tạo Feature Extractor: {feature_extractor_model}")
        feature_extractor = FeatureExtractor(ten_model=feature_extractor_model, thiet_bi=thiet_bi)

        # Bước 3: Trích xuất features
        ghi_log("Trích xuất features từ ảnh...")
        feature_matrix = trich_xuat_features(danh_sach_anh, feature_extractor)

        # Bước 4: Train Classifier
        ghi_log(f"Train Classifier: {loai_classifier}")
        classifier = PillClassifier(loai=loai_classifier, so_vong=so_vong)
        classifier.train(feature_matrix, np.array(danh_sach_nhan))

        # Bước 5: Lưu model
        duong_dan_luu = tim_file_trong_myenv(duong_dan_model)
        ghi_log(f"Lưu model vào: {duong_dan_luu}")
        classifier.luu_model(str(duong_dan_luu))

        # Thống kê
        cac_lop = np.unique(danh_sach_nhan)
        ket_qua = {
            "success": True,
            "total_images": len(danh_sach_anh),
            "total_classes": len(cac_lop),
            "classes": cac_lop.tolist(),
            "feature_dim": feature_matrix.shape[1],
            "classifier_type": loai_classifier,
            "feature_extractor": feature_extractor_model,
            "training_epochs": so_vong,
            "model_path": str(duong_dan_luu),
        }

        ghi_log("=" * 60)
        ghi_log("Huấn luyện hoàn tất!")
        ghi_log(f"Tổng ảnh: {ket_qua['total_images']}")
        ghi_log(f"Tổng lớp: {ket_qua['total_classes']}")
        ghi_log(f"Model đã lưu tại: {duong_dan_model}")
        ghi_log("=" * 60)

        return ket_qua

    except Exception as e:
        ghi_loi(f"Huấn luyện thất bại: {str(e)}", e)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    ket_qua = train_model()

    if ket_qua["success"]:
        print("\n✅ Huấn luyện thành công!")
        print(f"   Tổng ảnh: {ket_qua['total_images']}")
        print(f"   Tổng lớp: {ket_qua['total_classes']}")
        print(f"   Model: {ket_qua['model_path']}")
    else:
        print(f"\n❌ Huấn luyện thất bại: {ket_qua['error']}")