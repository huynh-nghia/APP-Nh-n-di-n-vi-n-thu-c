# 💊 Pill Recognition System

Nhận diện viên thuốc từ ảnh sử dụng AI.

## 🎯 Đây là gì?

- Upload ảnh viên thuốc → Hệ thống trả về tên thuốc
- Hỗ trợ 253 loại viên thuốc
- Có giao diện web dễ sử dụng
- Có API để tích hợp vào ứng dụng khác

## 🏗️ Hệ thống hoạt động như thế nào?

```
Ảnh → AI trích xuất đặc trưng → ML phân loại → Tên thuốc
```

**Tại sao dùng Hybrid AI + ML?**
- AI (Deep Learning): Đã học cách nhận diện hình ảnh từ hàng triệu ảnh
- ML (Machine Learning): Nhanh, chính xác, dễ train với ít dữ liệu

## 📁 Thư mục chứa gì?

```
myenv/
├── src/                    # Code chính
│   ├── path_utils.py      # Tìm file
│   ├── logger.py          # Ghi log
│   ├── utils.py           # Xử lý ảnh
│   ├── feature_extractor.py  # AI trích xuất đặc trưng
│   ├── classifier.py      # ML phân loại
│   ├── train.py           # Train model
│   ├── pill_label_utils.py   # Quản lý tên thuốc
│   ├── security.py        # Bảo mật
│   └── split_data.py      # Chia dataset
├── api/                    # Backend API
│   └── main.py            # API endpoints
├── app/                    # Giao diện web
│   └── ui.py              # Streamlit UI
├── models/                 # Lưu model đã train
├── logs/                   # File log
├── pill_labels.json        # Danh sách tên thuốc
├── requirements.txt        # Thư viện cần cài
├── run_api.sh              # Script chạy API
├── run_ui.sh               # Script chạy UI
└── .env                    # Cấu hình
```

## 🚀 Cách chạy nhanh


### (Lưu ý) nhớ tạo môi trường ảo trên máy mình python -m venv myenv (linux)

### Kích hoạt môi trường source myenv/bin/activate (linux)

### Cách 1: Dùng script (khuyên dùng)

**Chạy API:**
```bash
cd myenv
./run_api.sh
```

**Chạy UI (mở terminal khác):**
```bash
cd myenv
./run_ui.sh
```

### Cách 2: Chạy trực tiếp

**Chạy API:**
```bash
cd myenv
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Chạy UI:**
```bash
cd myenv
streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0
```

### Cách 3: Chạy thủ công

**Chạy API:**
```bash
cd myenv
python -m api.main
```

**Chạy UI:**
```bash
cd myenv
streamlit run app/ui.py
```

## 📋 Các bước đầy đủ

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Tạo file .env

Tạo file `.env` với nội dung:

```env
API_KEY=your-secret-api-key
BACKEND_URL=http://127.0.0.1:8000
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
ML_CLASSIFIER_MODEL_PATH=models/ml_classifier.pkl
MAX_FILE_SIZE_MB=5
CORS_ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
RATE_LIMIT_REQUESTS_PER_MINUTE=60
TRUST_X_FORWARDED_FOR=false
ENVIRONMENT=development
```

> ⚠️ **Quan trọng:** Không dùng API key mặc định/yếu. Hãy đặt chuỗi bí mật đủ mạnh.

### Bước 3: Chuẩn bị ảnh

Đặt ảnh vào thư mục `data/` với tên file theo định dạng:

```
(số_class)góc_chụp.jpg
```

Ví dụ:
- `(0)r15.jpg` - Class 0, góc 15 độ
- `(10)outline.jpg` - Class 10, góc outline
- `(252)r30.jpg` - Class 252, góc 30 độ

### Bước 4: Chia dataset (nếu muốn)

```bash
python -m src.split_data
```

Chia thành:
- Train: 70% (để train)
- Test: 15% (để test)
- Val: 15% (để validate)

### Bước 5: Train model

**Cách 1: Dùng giao diện web**
```bash
./run_ui.sh
```
Mở tab "🧠 Train model" → Nhấn "Bắt đầu train"

**Cách 2: Dùng command line**
```bash
python -m src.train
```

### Bước 6: Chạy hệ thống

**Terminal 1 - Chạy API:**
```bash
./run_api.sh
```

**Terminal 2 - Chạy UI:**
```bash
./run_ui.sh
```

### Bước 7: Sử dụng

- API chạy tại: `http://127.0.0.1:8000`
- UI chạy tại: `http://localhost:8501`

## 📡 API

### Kiểm tra trạng thái
```
GET /health
```

Nếu service chưa sẵn sàng (thiếu model hoặc lỗi startup), `/health` sẽ trả `503` với chi tiết `startup_error`.

### Nhận diện ảnh
```
POST /predict
Header: X-API-Key: your-api-key
Body: file ảnh (JPEG/PNG, max 5MB)
```

`/predict` có rate-limit theo IP (mặc định 60 req/phút, cấu hình bằng `RATE_LIMIT_REQUESTS_PER_MINUTE`).

Response:
```json
{
  "success": true,
  "message": "Dự đoán thành công",
  "predicted_pill": "2",
  "predicted_pill_display": "Paracetamol 500mg (ID: 2)",
  "confidence": 0.95,
  "probabilities": {
    "2": 0.95,
    "0": 0.03,
    "1": 0.02
  }
}
```

## 🎨 Giao diện web có gì?

### Tab 1: Nhận diện ảnh
- Upload ảnh đơn lẻ hoặc nhiều ảnh cùng lúc (Batch Prediction)
- Xem kết quả với biểu đồ Gauge Chart (độ tin cậy)
- Xem Top 5 loại thuốc có xác suất cao nhất (Horizontal Bar Chart)
- Lưu kết quả vào storage
- Xuất kết quả: ảnh đã nhận diện, JSON, CSV, ZIP

### Tab 2: Train model
- Cấu hình train (classifier, feature extractor, epochs)
- Theo dõi tiến trình
- Xem kết quả training

### Tab 3: Trực quan hóa
- Xem phân bố dữ liệu (Donut Chart, Bar Chart)
- Phân tích mất cân bằng dữ liệu (Class Imbalance)
- Heatmap phân bố dữ liệu (Class × Split)
- Thông số training (Accuracy, F1-Score, Radar Chart)
- Xem ảnh mẫu với metadata chi tiết

### Tab 4: Data Augmentation
- Tạo ảnh augmentation từ ảnh gốc
- 8 loại augmentation: rotate, flip, brightness, contrast, blur, noise, zoom
- Preview ảnh đã tạo
- Xuất ZIP tất cả ảnh augmentation

### Tab 5: Lịch sử dự đoán
- Xem lại các dự đoán đã thực hiện
- Lọc theo ngày và loại thuốc
- Biểu đồ xu hướng dự đoán theo ngày
- Thống kê tổng hợp

### Tab 6: Phát hiện chất lượng ảnh
- Đánh giá chất lượng ảnh trước khi nhận diện
- 5 chỉ số: độ mờ, độ sáng, độ tương phản, noise, kích thước
- Điểm tổng hợp 0-100 với đánh giá màu sắc
- Khuyến nghị cải thiện chất lượng

### Tab 7: Xuất báo cáo PDF
- Tạo báo cáo PDF chuyên nghiệp
- Bao gồm: tổng quan, bảng chi tiết, biểu đồ
- Cấu hình tiêu đề và số dòng tối đa
- Tải PDF trực tiếp

### Tab 8: Hướng dẫn
- Hướng dẫn sử dụng chi tiết
- Các tính năng nâng cao

## 🔧 Cấu hình

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| classifier_type | svm | svm hoặc random_forest |
| feature_extractor_model | mobilenetv2 | mobilenetv2 hoặc resnet18 |
| training_epochs | 1000 | Số vòng train |
| device | auto | auto, cpu, hoặc cuda |

## 📊 Log

Log được ghi vào:
- `logs/app.log` - Hoạt động thông thường
- `logs/error.log` - Lỗi
- `logs/security.log` - Bảo mật

## 🗄️ Lưu trữ lịch sử dự đoán

Lịch sử dự đoán trong UI hiện được lưu bằng **SQLite** tại:

- `data/predictions.db`

Ưu điểm so với JSON: an toàn hơn khi nhiều request đồng thời, dễ mở rộng và truy vấn thống kê.

## 🐛 Lỗi thường gặp

### Không kết nối được backend
- Kiểm tra backend đã chạy chưa
- Kiểm tra `BACKEND_URL` trong `.env`
- Nhấn nút "Kiểm tra Health Status" trong UI

### API Key sai
- Kiểm tra `API_KEY` trong `.env`
- Gửi đúng header `X-API-Key`

### Model chưa train
- Train model trước khi predict
- Kiểm tra file `models/ml_classifier.pkl` tồn tại

### Upload file lỗi
- Chỉ chấp nhận JPEG/PNG
- Tối đa 5MB
- File không bị hỏng

## 📝 License

Van Lang License

## 👥 Team

Pill Recognition Team
