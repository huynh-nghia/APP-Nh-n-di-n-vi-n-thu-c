# 🚀 Hướng dẫn triển khai lên Hugging Face Spaces

✅ Dự án đã được chuẩn bị sẵn sàng để triển khai lên HF Spaces chỉ với vài bước!

---

## 📋 Các file đã chuẩn bị:
- ✅ `Dockerfile` - Đã cấu hình đúng chuẩn HF Spaces
- ✅ `Dockerfile.api` + `Dockerfile.ui` (cho deployment riêng nếu cần)
- ✅ `docker-compose.yml`
- ✅ Tất cả dependencies trong requirements.txt đã được kiểm tra

---

## 🎯 Các bước triển khai trên Hugging Face:

### Bước 1: Tạo Space mới
1.  Truy cập https://huggingface.co/spaces
2.  Click **Create new Space**
3.  Điền thông tin:
    - Space name: `pill-recognition` (tên bất kỳ)
    - License: `MIT`
    - **Space SDK: `Docker`** ✅ **QUAN TRỌNG**
    - Hardware: `CPU basic` (2 vCPU / 16 GB RAM) - đủ chạy
    - Public / Private tùy chọn
4.  Click **Create Space**

### Bước 2: Upload code lên Space
Có 2 cách:

#### ✅ Cách 1: Push trực tiếp từ git
```bash
# Thêm remote HF
git remote add hf https://huggingface.co/spaces/[USERNAME]/pill-recognition

# Push toàn bộ code
git push hf main
```

#### ✅ Cách 2: Upload file thủ công
Tải toàn bộ thư mục dự án lên HF Spaces qua giao diện web.

---

## ⚙️ Cấu hình tùy chọn (nếu cần)

### Thêm biến môi trường trên HF:
Vào tab **Settings** của Space →  **Repository secrets** thêm:
```
API_KEY=your-super-secret-key-here
MAX_FILE_SIZE_MB=10
RATE_LIMIT_REQUESTS_PER_MINUTE=30
```

---

## 📊 Hardware Recommendation:
| Loại Hardware | Ưu điểm | Giá ước tính |
|---|---|---|
| CPU basic | Miễn phí, chạy đủ | $0 / giờ |
| CPU upgrade | 4 vCPU 32GB RAM | $0.03 / giờ |
| T4 GPU | Nhanh hơn nhiều | $0.60 / giờ |

> 💡 Chạy trên CPU đã đủ cho ứng dụng này (khoảng 0.5-1 giây / ảnh)

---

## ✅ Sau khi deploy xong:
- App sẽ tự build và chạy trong khoảng 5-10 phút
- Giao diện Web được truy cập tại: `https://[USERNAME]-pill-recognition.hf.space`
- API chạy nội bộ trong container
- Tất cả tính năng hoạt động y hệt local

---

## 🚨 Lưu ý quan trọng:
1.  Upload file `ml_classifier.pkl` đã train vào thư mục `myenv/models/` trước khi push
2.  Model ~ 100MB nên sẽ được load ngay khi app khởi động
3.  Mỗi khi push code mới HF sẽ tự động build lại container
4.  Không commit file `.env` lên git / HF

---

## 🔍 Kiểm tra lỗi:
Nếu app không chạy, xem log tại tab **Logs** trên HF Spaces.

---

✅ Dự án đã hoàn toàn chuẩn bị sẵn sàng cho Hugging Face Spaces!