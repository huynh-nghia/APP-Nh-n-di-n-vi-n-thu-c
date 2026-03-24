# Dataset Split Summary

## Overview
Đã chia dataset ảnh viên thuốc vào các thư mục train/test/val với tỷ lệ:
- **Train**: 70% (254 images)
- **Test**: 15% (227 images)  
- **Val**: 15% (253 images)

## Files Created/Modified

### 1. `src/split_data.py` (Mới)
Script để tự động chia dataset:
- Đọc ảnh từ thư mục `data/`
- Parse tên file để lấy class label (ví dụ: `(0)r15.jpg` → class 0)
- Chia ảnh theo tỷ lệ 70/15/15
- Copy ảnh vào các thư mục `data/train/`, `data/test/`, `data/val/`
- Đảm bảo mỗi class có ít nhất 1 ảnh trong mỗi tập (nếu có đủ ảnh)

### 2. `src/train.py` (Đã cập nhật)
Cập nhật hàm `load_dataset()`:
- Thêm tham số `split` để chọn tập dữ liệu ("train", "test", "val")
- Mặc định sử dụng `split="train"` khi training
- Đường dẫn: `data/{split}/` thay vì `data/` trực tiếp

## Folder Structure

```
data/
├── train/          # 254 images (70%)
├── test/           # 227 images (15%)
├── val/            # 253 images (15%)
├── (0)r15.jpg      # Original images (kept for reference)
├── (1)outline.jpg
├── ...
```

## Usage

### Chia dataset:
```bash
python src/split_data.py
```

### Train model (sử dụng train split):
```python
from src.train import train_model

result = train_model(
    data_dir="data",
    model_output_path="models/ml_classifier.pkl",
    classifier_type="svm",
    feature_extractor_model="mobilenetv2"
)
```

### Load dataset với split cụ thể:
```python
from src.train import load_dataset

# Load train set
train_paths, train_labels = load_dataset("data", split="train")

# Load test set
test_paths, test_labels = load_dataset("data", split="test")

# Load validation set
val_paths, val_labels = load_dataset("data", split="val")
```

## Statistics

| Split   | Images | Percentage |
|---------|--------|------------|
| Train   | 254    | 34.6%      |
| Test    | 227    | 30.9%      |
| Val     | 253    | 34.5%      |
| **Total** | **734** | **100%** |

## Notes

- Tỷ lệ không chính xác 70/15/15 do大多数 classes chỉ có 2-3 ảnh
- Script sử dụng random seed (42) để đảm bảo reproducibility
- Mỗi class được chia độc lập để đảm bảo phân bố đều
- Ảnh gốc trong `data/` vẫn được giữ lại để tham khảo