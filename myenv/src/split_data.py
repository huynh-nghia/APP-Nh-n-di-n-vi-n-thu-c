"""
File này chia dataset thành các thư mục train/test/val.

Mục đích:
    - Chia dữ liệu thành 3 tập: train (70%), test (15%), val (15%)
    - Đảm bảo mỗi class có đủ ảnh trong mỗi tập
    - Có thể chạy lại nhiều lần với cùng kết quả (reproducible)

Tại sao cần chia dataset?
    - Train: Dùng để huấn luyện model (70%)
    - Test: Đánh giá model sau khi train (15%)
    - Val: Điều chỉnh hyperparameters (15%)
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

from .path_utils import tim_file_data


def chia_dataset(
    thu_muc_nguon: str = "data",
    ty_le_train: float = 0.70,
    ty_le_test: float = 0.15,
    ty_le_val: float = 0.15,
    random_seed: int = 42,
) -> dict:
    """
    Chia dataset vào các thư mục train/test/val.
    
    Args:
        thu_muc_nguon: Thư mục chứa tất cả ảnh gốc
        ty_le_train: Tỷ lệ dữ liệu train (mặc định 70%)
        ty_le_test: Tỷ lệ dữ liệu test (mặc định 15%)
        ty_le_val: Tỷ lệ dữ liệu validation (mặc định 15%)
        random_seed: Seed để đảm bảo kết quả giống nhau mỗi lần chạy
    
    Returns:
        Dict chứa thống kê về quá trình chia
    """
    # Kiểm tra tổng tỷ lệ bằng 100%
    assert abs(ty_le_train + ty_le_test + ty_le_val - 1.0) < 1e-6, \
        "Tổng tỷ lệ phải bằng 1.0 (100%)"

    # Set random seed
    random.seed(random_seed)

    # Đường dẫn
    duong_dan_nguon = tim_file_data(thu_muc_nguon)

    if not duong_dan_nguon.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {thu_muc_nguon}")

    # Tạo thư mục đích
    duong_dan_train = duong_dan_nguon / "train"
    duong_dan_test = duong_dan_nguon / "test"
    duong_dan_val = duong_dan_nguon / "val"

    for duong_dan in [duong_dan_train, duong_dan_test, duong_dan_val]:
        duong_dan.mkdir(exist_ok=True)

    # Thu thập ảnh theo class
    anh_theo_lop = defaultdict(list)

    for file_path in sorted(duong_dan_nguon.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Parse filename để lấy class
                import re
                pattern = r'\((\d+)\)(r15|r30|outline)\.(jpg|jpeg|png)$'
                match = re.match(pattern, file_path.name, re.IGNORECASE)

                if match:
                    class_label = int(match.group(1))
                    anh_theo_lop[class_label].append(file_path)
            except:
                print(f"Bỏ qua file: {file_path.name}")

    # Chia dữ liệu cho mỗi class
    thong_ke = {
        "train": 0,
        "test": 0,
        "val": 0,
        "total_classes": len(anh_theo_lop),
    }

    for class_label, danh_sach_anh in sorted(anh_theo_lop.items()):
        # Shuffle images
        random.shuffle(danh_sach_anh)

        so_luong = len(danh_sach_anh)

        # Tính số lượng ảnh cho mỗi tập
        if so_luong >= 4:
            so_train = max(1, int(so_luong * ty_le_train))
            so_test = max(1, int(so_luong * ty_le_test))
            so_val = so_luong - so_train - so_test
            if so_val < 1:
                so_val = 1
                so_test = so_luong - so_train - so_val
        elif so_luong == 3:
            so_train = 1
            so_test = 1
            so_val = 1
        elif so_luong == 2:
            so_train = 1
            so_test = 0
            so_val = 1
        else:
            so_train = 1
            so_test = 0
            so_val = 0

        # Chia images
        anh_train = danh_sach_anh[:so_train]
        anh_test = danh_sach_anh[so_train:so_train + so_test]
        anh_val = danh_sach_anh[so_train + so_test:]

        # Copy files
        for img in anh_train:
            shutil.copy2(img, duong_dan_train / img.name)
            thong_ke["train"] += 1

        for img in anh_test:
            shutil.copy2(img, duong_dan_test / img.name)
            thong_ke["test"] += 1

        for img in anh_val:
            shutil.copy2(img, duong_dan_val / img.name)
            thong_ke["val"] += 1

        print(f"Class {class_label}: {so_luong} ảnh → "
              f"Train: {len(anh_train)}, Test: {len(anh_test)}, Val: {len(anh_val)}")

    # In thống kê
    tong = thong_ke["train"] + thong_ke["test"] + thong_ke["val"]
    print("\n" + "=" * 60)
    print("Thống kê chia Dataset")
    print("=" * 60)
    print(f"Tổng số class: {thong_ke['total_classes']}")
    print(f"Tổng số ảnh: {tong}")
    print(f"Train: {thong_ke['train']} ảnh ({thong_ke['train']/tong*100:.1f}%)")
    print(f"Test:  {thong_ke['test']} ảnh ({thong_ke['test']/tong*100:.1f}%)")
    print(f"Val:   {thong_ke['val']} ảnh ({thong_ke['val']/tong*100:.1f}%)")
    print("=" * 60)

    return thong_ke


if __name__ == "__main__":
    print("Bắt đầu chia dataset...")
    print("-" * 60)

    ket_qua = chia_dataset(
        thu_muc_nguon="data",
        ty_le_train=0.70,
        ty_le_test=0.15,
        ty_le_val=0.15,
        random_seed=42,
    )

    print("\n✅ Hoàn thành chia dataset!")