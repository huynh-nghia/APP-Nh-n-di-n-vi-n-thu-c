# ========================================
# 🐳 HƯỚNG DẪN DOCKER - Pill Recognition App
# ========================================
# File này giải thích cách sử dụng Docker cho ứng dụng nhận diện viên thuốc
# Dành cho người mới bắt đầu!

## 📖 MỤC LỤC
1. [Docker là gì?](#docker-là-gì)
2. [Kiến trúc ứng dụng](#kiến-trúc-ứng-dụng)
3. [Các file Docker](#các-file-docker)
4. [Cách sử dụng](#cách-sử-dụng)
5. [Giải thích chi tiết](#giải-thích-chi-tiết)
6. [Xử lý sự cố](#xử-lý-sự-cố)

---

## 🤔 Docker là gì?

**Docker** giống như một "hộp đựng" cho ứng dụng của bạn.

### Ví dụ dễ hiểu:
- **Trước Docker**: Ứng dụng chạy trên máy bạn, khi chuyển sang máy khác có thể lỗi
- **Sau Docker**: Ứng dụng được đóng gói trong "hộp", chạy được ở bất kỳ đâu

### Lợi ích:
✅ **Nhất quán**: Chạy giống nhau trên mọi máy  
✅ **Dễ deploy**: Chỉ cần 1 lệnh là chạy được  
✅ **Cô lập**: Không xung đột với ứng dụng khác  
✅ **Nhẹ hơn VM**: Không cần cài cả hệ điều hành  

---

## 🏗️ Kiến trúc ứng dụng

```
┌─────────────────────────────────────────────────────────┐
│                    MÁY TÍNH CỦA BẠN                      │
│                                                          │
│  ┌─────────────────┐         ┌─────────────────┐        │
│  │   UI Container   │────────▶│  API Container   │       │
│  │   (Streamlit)    │         │   (FastAPI)      │       │
│  │   Port: 8501     │         │   Port: 8000     │       │
│  └─────────────────┘         └─────────────────┘        │
│           │                           │                  │
│           └───────────────────────────┘                  │
│                    pill-network                          │
└─────────────────────────────────────────────────────────┘
```

### Giải thích:
- **UI Container**: Chạy giao diện web (Streamlit)
- **API Container**: Chạy backend xử lý ảnh (FastAPI)
- **pill-network**: Mạng nội bộ giúp 2 container giao tiếp

---

## 📁 Các file Docker

| File | Mục đích | Giải thích đơn giản |
|------|----------|---------------------|
| `Dockerfile.api` | Hướng dẫn build API | "Công thức nấu ăn" cho API |
| `Dockerfile.ui` | Hướng dẫn build UI | "Công thức nấu ăn" cho UI |
| `docker-compose.yml` | Điều phối containers | "Nhạc trưởng" chỉ huy cả dàn nhạc |
| `.dockerignore` | Loại bỏ file không cần | "Danh sách đen" file không copy |
| `DOCKER_README.md` | Hướng dẫn sử dụng | File bạn đang đọc đây! |

---

## 🚀 Cách sử dụng

### Bước 1: Cài đặt Docker

```bash
# Kiểm tra Docker đã cài chưa
docker --version

# Nếu chưa cài, tải từ: https://docs.docker.com/get-docker/
```

### Bước 2: Build images (lần đầu)

```bash
# Build cả 2 images (API + UI)
docker compose build

# Giải thích:
# - docker: Lệnh Docker
# - compose: Chạy nhiều container
# - build: Tạo images từ Dockerfiles
```

**Thời gian**: 5-10 phút (tùy mạng)  
**Kết quả**: Tạo 2 images: `pill_recognition_app-api` và `pill_recognition_app-ui`

### Bước 3: Chạy containers

```bash
# Chạy cả 2 containers (background)
docker compose up -d

# Giải thích:
# - up: Khởi động containers
# - -d: Chạy ngầm (detach)
```

**Kết quả**: 2 containers đang chạy

### Bước 4: Kiểm tra trạng thái

```bash
# Xem containers đang chạy
docker compose ps

# Kết quả mong đợi:
# NAME                   IMAGE                      STATUS
# pill-recognition-api   pill_recognition_app-api   Up (healthy)
# pill-recognition-ui    pill_recognition_app-ui    Up (healthy)
```

### Bước 5: Truy cập ứng dụng

Mở trình duyệt:
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Bước 6: Xem logs (nếu cần)

```bash
# Xem logs tất cả containers
docker compose logs -f

# Xem logs riêng API
docker compose logs -f api

# Xem logs riêng UI
docker compose logs -f ui
```

### Bước 7: Dừng containers

```bash
# Dừng containers (giữ data)
docker compose stop

# Dừng và xóa containers
docker compose down

# Dừng, xóa containers và volumes
docker compose down -v
```

---

## 📚 Giải thích chi tiết

### Dockerfile.api (11 bước)

```
BƯỚC 1: FROM python:3.10-slim
        → Lấy "nền" là Python 3.10 (bản nhẹ)

BƯỚC 2: WORKDIR /app
        → Tạo thư mục làm việc /app

BƯỚC 3: ENV PYTHONDONTWRITEBYTECODE=1
        → Cấu hình Python (không tạo file .pyc)

BƯỚC 4: RUN apt-get install ...
        → Cài thư viện hệ thống (OpenCV, libmagic, ...)

BƯỚC 5: COPY requirements.txt .
        → Copy file danh sách packages

BƯỚC 6: RUN pip install -r requirements.txt
        → Cài Python packages (FastAPI, PyTorch, ...)

BƯỚC 7: COPY myenv/api/ ./api/
        → Copy code ứng dụng

BƯỚC 8: RUN mkdir -p logs
        → Tạo thư mục logs

BƯỚC 9: EXPOSE 8000
        → Mở port 8000

BƯỚC 10: HEALTHCHECK ...
         → Cấu hình kiểm tra sức khỏe

BƯỚC 11: CMD ["uvicorn", ...]
         → Lệnh chạy khi container start
```

### docker-compose.yml (2 services)

```yaml
services:
  api:                    # Service 1: API
    build: ...            # Build từ Dockerfile.api
    ports: "8000:8000"    # Map port
    environment: ...      # Biến môi trường
    volumes: ...          # Mount thư mục
    healthcheck: ...      # Kiểm tra sức khỏe

  ui:                     # Service 2: UI
    build: ...            # Build từ Dockerfile.ui
    ports: "8501:8501"    # Map port
    depends_on: api       # Chờ API healthy
    healthcheck: ...      # Kiểm tra sức khỏe
```

### Volumes (Mount thư mục)

```yaml
volumes:
  - ./myenv/models:/app/models
```

**Giải thích**:
- `./myenv/models`: Thư mục trên máy bạn
- `:/app/models`: Thư mục trong container
- **Kết quả**: Container đọc file từ máy bạn

**Lý do**: Model file lớn (~100MB), không nên copy vào image

---

## 🔧 Xử lý sự cố

### Lỗi 1: Port bị chiếm

**Lỗi**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**Giải pháp**:
```bash
# Tìm process chiếm port
lsof -i :8000

# Hoặc đổi port trong docker-compose.yml
ports:
  - "8001:8000"  # Thay 8000 bằng 8001
```

### Lỗi 2: Container không start

**Kiểm tra**:
```bash
# Xem logs
docker compose logs api

# Xem trạng thái
docker compose ps
```

**Giải pháp**:
```bash
# Rebuild images
docker compose build --no-cache

# Restart containers
docker compose down && docker compose up -d
```

### Lỗi 3: Model không load

**Kiểm tra**:
```bash
# Kiểm tra file model tồn tại
ls -la myenv/models/ml_classifier.pkl

# Kiểm tra trong container
docker exec -it pill-recognition-api ls -la /app/models/
```

**Giải pháp**:
```bash
# Copy model vào container
docker cp myenv/models/ml_classifier.pkl pill-recognition-api:/app/models/

# Hoặc rebuild
docker compose build --no-cache
```

### Lỗi 4: Không truy cập được UI

**Kiểm tra**:
```bash
# Kiểm tra container UI
docker compose ps ui

# Kiểm tra health
curl http://localhost:8501/_stcore/health
```

**Giải pháp**:
```bash
# Restart UI
docker compose restart ui

# Xem logs
docker compose logs -f ui
```

---

## 🎯 Lệnh thường dùng

```bash
# Khởi động
docker compose up -d

# Dừng
docker compose down

# Xem trạng thái
docker compose ps

# Xem logs
docker compose logs -f

# Rebuild
docker compose build --no-cache

# Restart
docker compose restart

# Vào container
docker exec -it pill-recognition-api bash

# Xem resource usage
docker stats
```

---

## 📊 So sánh Docker vs Không Docker

| Tiêu chí | Không Docker | Có Docker |
|-----------|--------------|-----------|
| Cài đặt | Phải cài Python, packages | Chỉ cần Docker |
| Cấu hình | Thủ công | Tự động |
| Nhất quán | Có thể khác nhau | Giống hệt nhau |
| Deploy | Phức tạp | 1 lệnh |
| Cô lập | Có thể xung đột | Hoàn toàn cô lập |

---

## 🎓 Học thêm

- **Docker cơ bản**: https://docs.docker.com/get-started/
- **Docker Compose**: https://docs.docker.com/compose/
- **Dockerfile best practices**: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

---

## ✅ Checklist

Trước khi chạy:
- [ ] Docker đã cài đặt
- [ ] Docker daemon đang chạy
- [ ] Port 8000 và 8501 trống
- [ ] File `myenv/models/ml_classifier.pkl` tồn tại

Sau khi chạy:
- [ ] `docker compose ps` hiện 2 containers healthy
- [ ] Truy cập http://localhost:8501 thành công
- [ ] Truy cập http://localhost:8000/docs thành công
- [ ] Upload ảnh và nhận diện được

---

## 💡 Mẹo

1. **Lần đầu build chậm**: Do phải tải Python image và packages
2. **Lần sau nhanh hơn**: Docker cache layer
3. **Xem logs khi lỗi**: `docker compose logs -f`
4. **Rebuild khi thay đổi code**: `docker compose build --no-cache`
5. **Dùng `-d` để chạy ngầm**: `docker compose up -d`

---

**Chúc bạn thành công! 🎉**