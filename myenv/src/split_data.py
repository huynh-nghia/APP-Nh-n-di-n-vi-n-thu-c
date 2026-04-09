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

import random
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from .path_utils import tim_file_data


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r'\((\d+)\)(r15|r30|outline)\.(jpg|jpeg|png)$', re.IGNORECASE)


def _kiem_tra_ty_le(ty_le_train: float, ty_le_test: float, ty_le_val: float):
    """Kiểm tra các tỷ lệ đầu vào có hợp lệ hay không."""
    cac_ty_le = {
        "train": ty_le_train,
        "test": ty_le_test,
        "val": ty_le_val,
    }

    for ten_tap, ty_le in cac_ty_le.items():
        if ty_le < 0 or ty_le > 1:
            raise ValueError(f"Tỷ lệ {ten_tap} phải nằm trong khoảng từ 0.0 đến 1.0")

    tong_ty_le = ty_le_train + ty_le_test + ty_le_val
    if abs(tong_ty_le - 1.0) >= 1e-6:
        raise ValueError("Tổng tỷ lệ phải bằng 1.0 (100%)")


def _sap_xep_tap_theo_ty_le_giam_dan(ty_le_train: float, ty_le_test: float, ty_le_val: float) -> list:
    """Sắp xếp các tập theo tỷ lệ giảm dần, ưu tiên train khi bằng nhau."""
    cac_ty_le = {
        "train": ty_le_train,
        "test": ty_le_test,
        "val": ty_le_val,
    }
    muc_uu_tien = {"train": 0, "test": 1, "val": 2}
    return sorted(cac_ty_le.keys(), key=lambda tap: (-cac_ty_le[tap], muc_uu_tien[tap]))


def _sap_xep_tap_theo_ty_le_tang_dan(ty_le_train: float, ty_le_test: float, ty_le_val: float) -> list:
    """Sắp xếp các tập theo tỷ lệ tăng dần, ưu tiên giảm val/test trước train."""
    cac_ty_le = {
        "train": ty_le_train,
        "test": ty_le_test,
        "val": ty_le_val,
    }
    muc_uu_tien = {"val": 0, "test": 1, "train": 2}
    return sorted(cac_ty_le.keys(), key=lambda tap: (cac_ty_le[tap], muc_uu_tien[tap]))


def _tinh_so_luong_anh_cho_tung_tap(
    so_luong: int,
    ty_le_train: float,
    ty_le_test: float,
    ty_le_val: float,
) -> tuple:
    """
    Tính số ảnh cho train/test/val.

    Với class có rất ít ảnh, ưu tiên:
        - Luôn có ít nhất 1 ảnh train
        - Nếu có thể, phân bổ thêm cho test/val theo tỷ lệ đã chọn
    """
    if so_luong <= 0:
        return 0, 0, 0

    if so_luong == 1:
        return 1, 0, 0

    if so_luong in [2, 3]:
        so_train, so_test, so_val = 1, 0, 0
        con_lai = so_luong - 1

        for tap in _sap_xep_tap_theo_ty_le_giam_dan(ty_le_train, ty_le_test, ty_le_val):
            if con_lai == 0:
                break

            if tap == "train":
                continue

            ty_le = ty_le_test if tap == "test" else ty_le_val
            if ty_le > 0:
                if tap == "test":
                    so_test += 1
                else:
                    so_val += 1
                con_lai -= 1

        so_train += con_lai
        return so_train, so_test, so_val

    cac_ty_le = {
        "train": ty_le_train,
        "test": ty_le_test,
        "val": ty_le_val,
    }
    so_luong_toi_thieu = {
        "train": 1,
        "test": 1 if ty_le_test > 0 else 0,
        "val": 1 if ty_le_val > 0 else 0,
    }

    phan_bo = {
        "train": max(1, int(so_luong * ty_le_train)),
        "test": int(so_luong * ty_le_test) if ty_le_test > 0 else 0,
        "val": int(so_luong * ty_le_val) if ty_le_val > 0 else 0,
    }

    for tap, toi_thieu in so_luong_toi_thieu.items():
        phan_bo[tap] = max(phan_bo[tap], toi_thieu)

    while sum(phan_bo.values()) > so_luong:
        da_giam = False
        for tap in _sap_xep_tap_theo_ty_le_tang_dan(ty_le_train, ty_le_test, ty_le_val):
            if phan_bo[tap] > so_luong_toi_thieu[tap]:
                phan_bo[tap] -= 1
                da_giam = True
                break
        if not da_giam:
            break

    while sum(phan_bo.values()) < so_luong:
        for tap in _sap_xep_tap_theo_ty_le_giam_dan(ty_le_train, ty_le_test, ty_le_val):
            if cac_ty_le[tap] > 0 or tap == "train":
                phan_bo[tap] += 1
                break

    return phan_bo["train"], phan_bo["test"], phan_bo["val"]


def _thu_thap_anh_theo_lop(duong_dan_nguon: Path) -> defaultdict:
    """Thu thập danh sách ảnh gốc theo từng class."""
    anh_theo_lop = defaultdict(list)

    for file_path in sorted(duong_dan_nguon.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        match = FILENAME_PATTERN.match(file_path.name)
        if not match:
            continue

        class_label = int(match.group(1))
        anh_theo_lop[class_label].append(file_path)

    return anh_theo_lop


def _thu_thap_anh_theo_lop_tu_cac_thu_muc(cac_thu_muc: list) -> defaultdict:
    """Thu thập ảnh hợp lệ từ nhiều thư mục, tránh trùng tên file."""
    anh_theo_lop = defaultdict(list)
    ten_file_da_thay = set()

    for thu_muc in cac_thu_muc:
        if not thu_muc.exists() or not thu_muc.is_dir():
            continue

        for file_path in sorted(thu_muc.iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            if file_path.name in ten_file_da_thay:
                continue

            match = FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue

            class_label = int(match.group(1))
            anh_theo_lop[class_label].append(file_path)
            ten_file_da_thay.add(file_path.name)

    return anh_theo_lop


def _xac_dinh_nguon_anh_de_chia(duong_dan_nguon: Path) -> tuple:
    """Xác định nguồn ảnh dùng để chia: ưu tiên root, fallback sang train/test/val."""
    anh_theo_lop_goc = _thu_thap_anh_theo_lop(duong_dan_nguon)
    if anh_theo_lop_goc:
        return anh_theo_lop_goc, "root"

    cac_thu_muc_split = [
        duong_dan_nguon / tap
        for tap in ["train", "test", "val"]
        if (duong_dan_nguon / tap).exists()
    ]
    anh_theo_lop_split = _thu_thap_anh_theo_lop_tu_cac_thu_muc(cac_thu_muc_split)
    if anh_theo_lop_split:
        return anh_theo_lop_split, "split"

    return defaultdict(list), "empty"


def _dua_anh_nguon_vao_thu_muc_tam(anh_theo_lop: defaultdict, thu_muc_tam: Path) -> defaultdict:
    """Sao chép ảnh nguồn vào thư mục tạm để có thể chia lại an toàn."""
    anh_theo_lop_tam = defaultdict(list)

    for class_label, danh_sach_anh in sorted(anh_theo_lop.items()):
        for file_path in danh_sach_anh:
            duong_dan_tam = thu_muc_tam / file_path.name
            shutil.copy2(file_path, duong_dan_tam)
            anh_theo_lop_tam[class_label].append(duong_dan_tam)

    return anh_theo_lop_tam


def _chuan_bi_thu_muc_split(cac_thu_muc_split: list, xoa_split_cu: bool):
    """Tạo mới hoặc làm sạch các thư mục train/test/val."""
    for duong_dan in cac_thu_muc_split:
        if xoa_split_cu and duong_dan.exists():
            shutil.rmtree(duong_dan)
        duong_dan.mkdir(parents=True, exist_ok=True)


def _thuc_hien_chia_dataset(
    anh_theo_lop: defaultdict,
    duong_dan_train: Path,
    duong_dan_test: Path,
    duong_dan_val: Path,
    ty_le_train: float,
    ty_le_test: float,
    ty_le_val: float,
    bo_sinh_ngau_nhien: random.Random,
    thong_ke: dict,
    verbose: bool,
):
    """Thực hiện chia dataset sau khi đã có danh sách ảnh nguồn hợp lệ."""
    for class_label, danh_sach_anh in sorted(anh_theo_lop.items()):
        # Shuffle images
        danh_sach_anh = list(danh_sach_anh)
        bo_sinh_ngau_nhien.shuffle(danh_sach_anh)

        so_luong = len(danh_sach_anh)

        # Tính số lượng ảnh cho mỗi tập
        so_train, so_test, so_val = _tinh_so_luong_anh_cho_tung_tap(
            so_luong=so_luong,
            ty_le_train=ty_le_train,
            ty_le_test=ty_le_test,
            ty_le_val=ty_le_val,
        )

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

        thong_ke["class_details"].append(
            {
                "class_label": class_label,
                "total": so_luong,
                "train": len(anh_train),
                "test": len(anh_test),
                "val": len(anh_val),
            }
        )

        if verbose:
            print(
                f"Class {class_label}: {so_luong} ảnh → "
                f"Train: {len(anh_train)}, Test: {len(anh_test)}, Val: {len(anh_val)}"
            )


def chia_dataset(
    thu_muc_nguon: str = "data",
    ty_le_train: float = 0.70,
    ty_le_test: float = 0.15,
    ty_le_val: float = 0.15,
    random_seed: int = 42,
    xoa_split_cu: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Chia dataset vào các thư mục train/test/val.
    
    Args:
        thu_muc_nguon: Thư mục chứa tất cả ảnh gốc
        ty_le_train: Tỷ lệ dữ liệu train (mặc định 70%)
        ty_le_test: Tỷ lệ dữ liệu test (mặc định 15%)
        ty_le_val: Tỷ lệ dữ liệu validation (mặc định 15%)
        random_seed: Seed để đảm bảo kết quả giống nhau mỗi lần chạy
        xoa_split_cu: Nếu True, xóa thư mục train/test/val cũ trước khi chia lại
        verbose: Nếu True, in thống kê ra màn hình
    
    Returns:
        Dict chứa thống kê về quá trình chia
    """
    _kiem_tra_ty_le(ty_le_train, ty_le_test, ty_le_val)

    bo_sinh_ngau_nhien = random.Random(random_seed)

    # Đường dẫn
    duong_dan_nguon = tim_file_data(thu_muc_nguon)

    if not duong_dan_nguon.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {thu_muc_nguon}")

    if not duong_dan_nguon.is_dir():
        raise NotADirectoryError(f"Đường dẫn nguồn không phải thư mục: {duong_dan_nguon}")

    da_co_split = any((duong_dan_nguon / tap).exists() for tap in ["train", "test", "val"])

    # Thu thập ảnh nguồn dùng để chia: ưu tiên root, fallback sang các split hiện có
    anh_theo_lop, che_do_nguon = _xac_dinh_nguon_anh_de_chia(duong_dan_nguon)

    if not anh_theo_lop:
        if da_co_split:
            raise ValueError(
                "Không tìm thấy ảnh hợp lệ trong thư mục gốc hoặc trong các thư mục train/test/val hiện có."
            )
        raise ValueError(f"Không tìm thấy ảnh hợp lệ trong thư mục nguồn: {duong_dan_nguon}")

    # Tạo thư mục đích
    duong_dan_train = duong_dan_nguon / "train"
    duong_dan_test = duong_dan_nguon / "test"
    duong_dan_val = duong_dan_nguon / "val"

    # Chia dữ liệu cho mỗi class
    xoa_split_cu_thuc_te = xoa_split_cu or che_do_nguon == "split"
    thong_ke = {
        "train": 0,
        "test": 0,
        "val": 0,
        "total_images": sum(len(ds) for ds in anh_theo_lop.values()),
        "total_classes": len(anh_theo_lop),
        "source_path": str(duong_dan_nguon),
        "source_mode": che_do_nguon,
        "random_seed": random_seed,
        "ratios": {
            "train": ty_le_train,
            "test": ty_le_test,
            "val": ty_le_val,
        },
        "xoa_split_cu": xoa_split_cu,
        "xoa_split_cu_thuc_te": xoa_split_cu_thuc_te,
        "class_details": [],
    }

    if che_do_nguon == "split":
        with tempfile.TemporaryDirectory() as thu_muc_tam:
            anh_theo_lop_tam = _dua_anh_nguon_vao_thu_muc_tam(
                anh_theo_lop,
                Path(thu_muc_tam),
            )
            _chuan_bi_thu_muc_split(
                [duong_dan_train, duong_dan_test, duong_dan_val],
                xoa_split_cu=xoa_split_cu_thuc_te,
            )
            _thuc_hien_chia_dataset(
                anh_theo_lop=anh_theo_lop_tam,
                duong_dan_train=duong_dan_train,
                duong_dan_test=duong_dan_test,
                duong_dan_val=duong_dan_val,
                ty_le_train=ty_le_train,
                ty_le_test=ty_le_test,
                ty_le_val=ty_le_val,
                bo_sinh_ngau_nhien=bo_sinh_ngau_nhien,
                thong_ke=thong_ke,
                verbose=verbose,
            )
    else:
        _chuan_bi_thu_muc_split(
            [duong_dan_train, duong_dan_test, duong_dan_val],
            xoa_split_cu=xoa_split_cu_thuc_te,
        )
        _thuc_hien_chia_dataset(
            anh_theo_lop=anh_theo_lop,
            duong_dan_train=duong_dan_train,
            duong_dan_test=duong_dan_test,
            duong_dan_val=duong_dan_val,
            ty_le_train=ty_le_train,
            ty_le_test=ty_le_test,
            ty_le_val=ty_le_val,
            bo_sinh_ngau_nhien=bo_sinh_ngau_nhien,
            thong_ke=thong_ke,
            verbose=verbose,
        )

    # In thống kê
    tong = thong_ke["train"] + thong_ke["test"] + thong_ke["val"]
    if tong == 0:
        raise ValueError("Không thể chia dataset vì không có ảnh nào được phân bổ vào train/test/val")

    if verbose:
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
        xoa_split_cu=True,
        verbose=True,
    )

    print("\n✅ Hoàn thành chia dataset!")