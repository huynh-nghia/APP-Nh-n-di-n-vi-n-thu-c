# Dockerfile cho Hugging Face Spaces
FROM python:3.11-slim

WORKDIR /app

# Cài đặt hệ thống dependencies
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# Copy toàn bộ dự án
COPY . .

# Cài đặt python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r myenv/requirements.txt

# Tạo thư mục cần thiết
RUN mkdir -p myenv/models myenv/logs myenv/data

# Expose port 7860 (port mặc định của HF Spaces)
EXPOSE 7860

# Environment variables cho production
ENV ENVIRONMENT=production
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV BACKEND_URL=http://localhost:8000
ENV API_KEY=hf-space-auto-generated-key-${RANDOM}
ENV TRUST_X_FORWARDED_FOR=true

# Script chạy cả 2 service cùng lúc
CMD ["bash", "-c", "\
    cd myenv && \
    uvicorn api.main:app --host 0.0.0.0 --port 8000 & \
    sleep 5 && \
    streamlit run app/ui.py --server.port 7860 --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false --browser.gatherUsageStats false\
"]