"""
File này tạo giao diện web cho Pill Recognition App.

Mục đích:
    - Giao diện thân thiện để upload ảnh viên thuốc
    - Hiển thị kết quả nhận diện (tên thuốc, độ tin cậy, xác suất)
    - Train model trực tiếp từ giao diện
    - Trực quan hóa dataset

Cách sử dụng:
    streamlit run app/ui.py
"""

import streamlit as st
import requests
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# Load environment variables
load_dotenv()

try:
    from src.path_utils import tim_file_trong_myenv, tim_file_data
    from src.pill_label_utils import (
        load_mapping,
        luu_mapping,
        lay_ten_thuoc,
        kiem_tra_co_ten_thuoc,
        format_hien_thi,
        chuan_hoa_label,
        FILE_MAPPING,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.path_utils import tim_file_trong_myenv, tim_file_data
    from src.pill_label_utils import (
        load_mapping,
        luu_mapping,
        lay_ten_thuoc,
        kiem_tra_co_ten_thuoc,
        format_hien_thi,
        chuan_hoa_label,
        FILE_MAPPING,
    )

# ========================================
# Cấu hình trang
# ========================================
st.set_page_config(
    page_title="Pill Recognition App",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================================
# Constants
# ========================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "default-api-key-change-me")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 5))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"\((\d+)\)(r15|r30|outline)\.(jpg|jpeg|png)$", re.IGNORECASE)


def dem_anh_trong_thu_muc(thu_muc: Path) -> int:
    """Đếm số ảnh hợp lệ trong thư mục."""
    if not thu_muc.exists() or not thu_muc.is_dir():
        return 0
    return sum(
        1 for f in thu_muc.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    )


def parse_metadata(ten_file: str) -> dict:
    """Parse metadata từ tên file."""
    match = FILENAME_PATTERN.match(ten_file)
    if not match:
        return {"class_label": "unknown", "class_sort": 10**9, "view_type": "unknown"}
    return {
        "class_label": match.group(1),
        "class_sort": int(match.group(1)),
        "view_type": match.group(2).lower(),
    }


@st.cache_data(show_spinner=False)
def load_mapping_cached() -> dict:
    """Cached wrapper for load_mapping."""
    return load_mapping()


def ve_nhan_len_anh(image: Image.Image, primary: str, secondary: str = None) -> Image.Image:
    """Vẽ nhãn lên ảnh."""
    labeled = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", labeled.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    lines = [primary]
    if secondary:
        lines.append(secondary)

    margin = max(8, min(labeled.size) // 24)
    padding = max(6, margin // 2)

    # Tính kích thước text
    max_width = 0
    total_height = 0
    line_metrics = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        line_metrics.append((line, w, h))
        max_width = max(max_width, w)
        total_height += h

    total_height += 4 * (len(lines) - 1)

    # Vẽ background
    draw.rounded_rectangle(
        (margin, margin, margin + max_width + padding * 2, margin + total_height + padding * 2),
        radius=max(8, padding),
        fill=(0, 0, 0, 175),
        outline=(255, 255, 255, 120),
        width=1,
    )

    # Vẽ text
    y = margin + padding
    for line, _, h in line_metrics:
        draw.text((margin + padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += h + 4

    return Image.alpha_composite(labeled, overlay).convert("RGB")


@st.cache_data(show_spinner=False)
def scan_dataset(duong_dan: str) -> list:
    """Quét dataset."""
    resolved = tim_file_data(duong_dan)
    if not resolved.exists() or not resolved.is_dir():
        return []

    splits = ["train", "test", "val"]
    targets = [(s, resolved / s) for s in splits if (resolved / s).exists()]
    if not targets:
        targets = [("root", resolved)]

    records = []
    for split_name, target_dir in targets:
        for f in sorted(target_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
                meta = parse_metadata(f.name)
                records.append({
                    "split": split_name,
                    "filename": f.name,
                    "filepath": str(f),
                    "class_label": meta["class_label"],
                    "class_sort": meta["class_sort"],
                    "view_type": meta["view_type"],
                    "size_kb": round(f.stat().st_size / 1024, 2),
                })
    return records


def get_dataset_summary(duong_dan: str) -> dict:
    """Lấy thông tin tổng quan dataset."""
    resolved = tim_file_data(duong_dan)
    splits = ["train", "test", "val"]
    summary = {
        "resolved_path": resolved,
        "exists": resolved.exists(),
        "is_dir": resolved.is_dir() if resolved.exists() else False,
        "root_images": 0,
        "split_counts": {s: 0 for s in splits},
        "has_split_dirs": False,
    }
    if summary["exists"] and summary["is_dir"]:
        summary["root_images"] = dem_anh_trong_thu_muc(resolved)
        summary["has_split_dirs"] = any((resolved / s).exists() for s in splits)
        for s in splits:
            summary["split_counts"][s] = dem_anh_trong_thu_muc(resolved / s)
    return summary


def get_model_summary(duong_dan: str) -> dict:
    """Lấy thông tin file model."""
    resolved = tim_file_trong_myenv(duong_dan)
    exists = resolved.exists() and resolved.is_file()
    return {
        "resolved_path": resolved,
        "exists": exists,
        "size_mb": (resolved.stat().st_size / (1024 * 1024)) if exists else 0,
        "updated_at": datetime.fromtimestamp(resolved.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S") if exists else None,
    }


def run_training(data_dir, model_path, classifier_type, fe_model, device_option, epochs):
    """Train model."""
    try:
        from src.train import train_model
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.train import train_model

    device = None if device_option == "auto" else device_option
    return train_model(
        duong_dan_data=data_dir,
        duong_dan_model=model_path,
        loai_classifier=classifier_type,
        feature_extractor_model=fe_model,
        thiet_bi=device,
        so_vong=epochs,
    )


# ========================================
# CSS
# ========================================
st.markdown("""
<style>
.header-title { color: #1f77e4; text-align: center; margin-bottom: 30px; }
.success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745; margin: 10px 0; }
.error-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; margin: 10px 0; }
.warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ffc107; margin: 10px 0; }
.info-box { background-color: #e7f3ff; padding: 15px; border-radius: 5px; border-left: 4px solid #0275d8; margin: 10px 0; }
.confidence-bar { width: 100%; height: 30px; background-color: #e9ecef; border-radius: 5px; overflow: hidden; }
.confidence-fill { height: 100%; background-color: #28a745; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========================================
# Header
# ========================================
st.markdown('<h1 class="header-title">💊 Pill Recognition System</h1>', unsafe_allow_html=True)
st.markdown("**Công nghệ Hybrid Deep Learning + Machine Learning**\n\nUpload ảnh viên thuốc để nhận diện hoặc train model mới.")

mapping = load_mapping_cached()

# ========================================
# Sidebar
# ========================================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.subheader("Kết nối Backend")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)
    api_key = st.text_input("API Key", value=API_KEY, type="password")

    st.subheader("Kiểm tra Kết nối")
    if st.button("🔍 Kiểm tra Health Status"):
        try:
            resp = requests.get(f"{backend_url}/health", timeout=5)
            if resp.status_code == 200:
                st.success("✅ Backend disponible!")
                st.json(resp.json())
            else:
                st.error(f"❌ Status code: {resp.status_code}")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

    with st.expander("📋 Thông tin Debug"):
        st.code(f"Backend URL: {backend_url}\nAPI Key: {'*' * len(api_key)}")
        st.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")

# ========================================
# Main Content
# ========================================
tab1, tab2, tab3, tab4 = st.tabs(["📸 Nhận diện ảnh", "🧠 Train model", "📊 Trực quan hóa", "📖 Hướng dẫn"])

# Tab 1: Nhận diện
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload Ảnh")
        uploaded_file = st.file_uploader("Chọn file ảnh (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.markdown(f'<div class="error-box">❌ File quá lớn: {file_size_mb:.2f}MB (tối đa {MAX_FILE_SIZE_MB}MB)</div>', unsafe_allow_html=True)
            else:
                image_bytes = uploaded_file.getvalue()
                if not image_bytes:
                    st.markdown('<div class="error-box">❌ File upload rỗng</div>', unsafe_allow_html=True)
                else:
                    image = Image.open(BytesIO(image_bytes))
                    st.image(image, caption=f"Ảnh upload: {uploaded_file.name}", width="stretch")

                    if st.button("🚀 Nhận diện", key="predict_button"):
                        with st.spinner("⏳ Đang xử lý..."):
                            try:
                                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type or "application/octet-stream")}
                                headers = {"X-API-Key": api_key}
                                resp = requests.post(f"{backend_url}/predict", files=files, headers=headers, timeout=30)

                                if resp.status_code == 200:
                                    result = resp.json()
                                    st.session_state.prediction_result = result
                                    if result.get("success"):
                                        st.markdown('<div class="success-box">✅ Nhận diện thành công!</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="error-box">❌ {result.get("error", "Unknown")}</div>', unsafe_allow_html=True)
                                else:
                                    st.session_state.pop("prediction_result", None)
                                    if resp.status_code == 401:
                                        st.markdown('<div class="error-box">❌ API Key không hợp lệ</div>', unsafe_allow_html=True)
                                    elif resp.status_code == 413:
                                        st.markdown('<div class="error-box">❌ File quá lớn</div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="error-box">❌ Lỗi server: {resp.status_code}</div>', unsafe_allow_html=True)
                            except Exception as e:
                                st.session_state.pop("prediction_result", None)
                                st.markdown(f'<div class="error-box">❌ Lỗi: {str(e)}</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Kết quả")
        if "prediction_result" in st.session_state:
            result = st.session_state.prediction_result
            if result.get("success"):
                predicted_label = result.get("predicted_pill", "unknown")
                predicted_display = result.get("predicted_pill_display") or format_hien_thi(predicted_label, mapping)

                st.markdown(f"**Viên thuốc nhận diện:** `{predicted_display}`")

                confidence = result.get("confidence", 0)
                st.markdown(f"**Độ tin cậy:** {confidence:.2%}")

                color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.5 else "#dc3545"
                st.markdown(f'<div class="confidence-bar"><div class="confidence-fill" style="width: {confidence*100}%; background-color: {color};">{confidence:.2%}</div></div>', unsafe_allow_html=True)

                st.markdown("**Xác suất cho các lớp:**")
                display_probs = result.get("display_probabilities")
                probs = display_probs or result.get("probabilities", {})
                if probs:
                    sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
                    for label, prob in sorted_probs.items():
                        display = label if display_probs else format_hien_thi(label, mapping)
                        st.write(f"🔹 {display}: {prob:.4f} ({prob*100:.2f}%)")
                        st.progress(prob)

# Tab 2: Train model
with tab2:
    st.subheader("🧠 Huấn luyện model")
    st.markdown("Train model trực tiếp từ Streamlit.")

    with st.form("training_form"):
        c1, c2 = st.columns(2)
        with c1:
            data_dir = st.text_input("Thư mục dữ liệu", value="data")
            classifier_type = st.selectbox("Classifier", ["svm", "random_forest"])
            epochs = st.number_input("Epochs / max_iter", min_value=1, value=1000, step=10)
            device_option = st.selectbox("Thiết bị", ["auto", "cpu", "cuda"])
        with c2:
            model_path = st.text_input("Đường dẫn lưu model", value="models/ml_classifier.pkl")
            fe_model = st.selectbox("Feature Extractor", ["mobilenetv2", "resnet18"])

        start_training = st.form_submit_button("🚀 Bắt đầu train", type="primary", width="stretch")

    dataset_summary = get_dataset_summary(data_dir)
    model_summary = get_model_summary(model_path)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("### 📂 Dataset")
        st.code(str(dataset_summary["resolved_path"]))
        if not dataset_summary["exists"]:
            st.error("Không tìm thấy thư mục dataset.")
        else:
            st.success("Dataset hợp lệ.")
            mc = st.columns(4)
            mc[0].metric("Ảnh gốc", dataset_summary["root_images"])
            mc[1].metric("Train", dataset_summary["split_counts"]["train"])
            mc[2].metric("Test", dataset_summary["split_counts"]["test"])
            mc[3].metric("Val", dataset_summary["split_counts"]["val"])

    with sc2:
        st.markdown("### 💾 Model Output")
        st.code(str(model_summary["resolved_path"]))
        if model_summary["exists"]:
            st.success("Đã có file model.")
            st.write(f"- Kích thước: {model_summary['size_mb']:.2f} MB")
            st.write(f"- Cập nhật: {model_summary['updated_at']}")
        else:
            st.info("Model sẽ được tạo mới sau khi train.")

    if start_training:
        total = dataset_summary["root_images"] + sum(dataset_summary["split_counts"].values())
        if not dataset_summary["exists"]:
            st.error("❌ Thư mục dataset không hợp lệ.")
        elif total == 0:
            st.error("❌ Không tìm thấy ảnh hợp lệ.")
        else:
            with st.spinner("⏳ Đang train..."):
                result = run_training(data_dir, model_path, classifier_type, fe_model, device_option, int(epochs))
            st.session_state.training_result = result
            if result.get("success"):
                st.success("✅ Train thành công!")
            else:
                st.error(f"❌ Train thất bại: {result.get('error')}")

    if "training_result" in st.session_state:
        result = st.session_state.training_result
        st.markdown("### 📈 Kết quả training")
        if result.get("success"):
            rc = st.columns(4)
            rc[0].metric("Tổng ảnh", result.get("total_images", 0))
            rc[1].metric("Số lớp", result.get("total_classes", 0))
            rc[2].metric("Feature dim", result.get("feature_dim", 0))
            rc[3].metric("Classifier", result.get("classifier_type", "-"))

# Tab 3: Trực quan hóa
with tab3:
    st.subheader("📊 Trực quan hóa dữ liệu")

    viz_dir = st.text_input("Thư mục dữ liệu", value="data")
    if st.button("🔄 Làm mới"):
        scan_dataset.clear()
        st.rerun()

    viz_summary = get_dataset_summary(viz_dir)
    viz_records = scan_dataset(viz_dir)

    if not viz_records:
        st.warning("Chưa tìm thấy ảnh hợp lệ.")
    else:
        df = pd.DataFrame(viz_records)
        df["pill_name"] = df["class_label"].apply(lambda v: lay_ten_thuoc(v, mapping))
        df["display_label"] = df["class_label"].apply(lambda v: format_hien_thi(v, mapping))

        valid_classes = df[df["class_label"] != "unknown"]["class_label"].nunique()
        valid_views = df[df["view_type"] != "unknown"]["view_type"].nunique()

        sc = st.columns(4)
        sc[0].metric("Tổng ảnh", len(df))
        sc[1].metric("Số class", valid_classes)
        sc[2].metric("Số góc", valid_views)
        sc[3].metric("Kích thước TB", f"{df['size_kb'].mean():.1f} KB")

        # Phân bố theo split
        st.markdown("#### Phân bố theo split")
        split_df = df.groupby("split").size().reset_index(name="count")
        st.bar_chart(split_df.set_index("split"))

        # Phân bố theo góc
        st.markdown("#### Phân bố theo góc chụp")
        view_df = df.groupby("view_type").size().reset_index(name="count")
        st.bar_chart(view_df.set_index("view_type"))

        # Metadata chi tiết
        st.markdown("#### 📋 Metadata chi tiết")
        
        # Tính toán metadata
        metadata_records = []
        for record in viz_records:
            try:
                img = Image.open(record["filepath"])
                metadata_records.append({
                    "filename": record["filename"],
                    "class_label": record["class_label"],
                    "pill_name": lay_ten_thuoc(record["class_label"], mapping),
                    "view_type": record["view_type"],
                    "split": record["split"],
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "size_kb": record["size_kb"],
                    "aspect_ratio": round(img.width / img.height, 2) if img.height > 0 else 0,
                })
            except Exception as e:
                metadata_records.append({
                    "filename": record["filename"],
                    "class_label": record["class_label"],
                    "pill_name": lay_ten_thuoc(record["class_label"], mapping),
                    "view_type": record["view_type"],
                    "split": record["split"],
                    "width": "N/A",
                    "height": "N/A",
                    "mode": "N/A",
                    "format": "N/A",
                    "size_kb": record["size_kb"],
                    "aspect_ratio": "N/A",
                })
        
        metadata_df = pd.DataFrame(metadata_records)
        
        # Hiển thị thống kê metadata
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
        
        with meta_col1:
            valid_dims = metadata_df[metadata_df["width"] != "N/A"]
            if len(valid_dims) > 0:
                avg_width = valid_dims["width"].mean()
                avg_height = valid_dims["height"].mean()
                st.metric("Kích thước TB", f"{avg_width:.0f} x {avg_height:.0f} px")
            else:
                st.metric("Kích thước TB", "N/A")
        
        with meta_col2:
            valid_modes = metadata_df[metadata_df["mode"] != "N/A"]
            if len(valid_modes) > 0:
                most_common_mode = valid_modes["mode"].mode()[0] if len(valid_modes["mode"].mode()) > 0 else "N/A"
                st.metric("Color Mode phổ biến", most_common_mode)
            else:
                st.metric("Color Mode phổ biến", "N/A")
        
        with meta_col3:
            valid_formats = metadata_df[metadata_df["format"] != "N/A"]
            if len(valid_formats) > 0:
                most_common_format = valid_formats["format"].mode()[0] if len(valid_formats["format"].mode()) > 0 else "N/A"
                st.metric("Format phổ biến", most_common_format)
            else:
                st.metric("Format phổ biến", "N/A")
        
        with meta_col4:
            valid_ratios = metadata_df[metadata_df["aspect_ratio"] != "N/A"]
            if len(valid_ratios) > 0:
                avg_ratio = valid_ratios["aspect_ratio"].mean()
                st.metric("Tỷ lệ khung hình TB", f"{avg_ratio:.2f}")
            else:
                st.metric("Tỷ lệ khung hình TB", "N/A")
        
        # Bảng metadata đầy đủ
        with st.expander("📊 Xem bảng metadata đầy đủ"):
            st.dataframe(
                metadata_df[["filename", "pill_name", "view_type", "split", "width", "height", "mode", "format", "size_kb", "aspect_ratio"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "filename": "Tên file",
                    "pill_name": "Tên thuốc",
                    "view_type": "Góc chụp",
                    "split": "Split",
                    "width": "Chiều rộng (px)",
                    "height": "Chiều cao (px)",
                    "mode": "Color Mode",
                    "format": "Format",
                    "size_kb": "Kích thước (KB)",
                    "aspect_ratio": "Tỷ lệ khung hình",
                }
            )
        
        # Phân bố kích thước ảnh
        st.markdown("#### 📐 Phân bố kích thước ảnh")
        valid_dims_df = metadata_df[metadata_df["width"] != "N/A"].copy()
        
        if len(valid_dims_df) > 0:
            dim_col1, dim_col2 = st.columns(2)
            
            with dim_col1:
                st.markdown("**Chiều rộng**")
                width_stats = valid_dims_df["width"].describe()
                st.write(f"- Min: {width_stats['min']:.0f} px")
                st.write(f"- Max: {width_stats['max']:.0f} px")
                st.write(f"- Mean: {width_stats['mean']:.0f} px")
                st.write(f"- Std: {width_stats['std']:.0f} px")
            
            with dim_col2:
                st.markdown("**Chiều cao**")
                height_stats = valid_dims_df["height"].describe()
                st.write(f"- Min: {height_stats['min']:.0f} px")
                st.write(f"- Max: {height_stats['max']:.0f} px")
                st.write(f"- Mean: {height_stats['mean']:.0f} px")
                st.write(f"- Std: {height_stats['std']:.0f} px")
            
            # Histogram kích thước
            st.markdown("**Phân bố tỷ lệ khung hình**")
            ratio_counts = valid_dims_df["aspect_ratio"].round(1).value_counts().sort_index()
            st.bar_chart(ratio_counts)
        else:
            st.info("Không có dữ liệu kích thước hợp lệ.")
        
        # Xem ảnh mẫu
        st.markdown("#### 🖼️ Xem ảnh mẫu")
        preview_split = st.selectbox("Lọc theo split", ["all"] + sorted(df["split"].unique().tolist()))
        preview_class = st.selectbox("Lọc theo class", ["all"] + sorted(df["class_label"].unique().tolist(), key=lambda v: (v == "unknown", int(v) if v.isdigit() else 10**9, v)))

        preview_df = df.copy()
        if preview_split != "all":
            preview_df = preview_df[preview_df["split"] == preview_split]
        if preview_class != "all":
            preview_df = preview_df[preview_df["class_label"] == preview_class]

        preview_df = preview_df.sort_values(["split", "class_sort", "filename"])
        st.caption(f"Tổng số ảnh: {len(preview_df)}")

        cols = st.columns(4)
        for i, record in enumerate(preview_df.head(20).to_dict("records")):
            with cols[i % 4]:
                try:
                    img = Image.open(record["filepath"])
                    img.thumbnail((512, 512))
                    primary = lay_ten_thuoc(record["class_label"], mapping)
                    secondary = f"ID: {record['class_label']} | {record['view_type']} | {record['split']}"
                    labeled = ve_nhan_len_anh(img, primary, secondary)
                    st.image(labeled, caption=record["filename"], width="stretch")
                except Exception as e:
                    st.warning(f"Không thể mở ảnh: {e}")

# Tab 4: Hướng dẫn
with tab4:
    st.subheader("📖 Hướng dẫn Sử dụng")
    st.markdown("""
    ### Các Bước Sử Dụng:
    
    1. **Upload Ảnh**: Chọn file ảnh viên thuốc (JPG, JPEG, PNG, max 5MB)
    2. **Nhận Diện**: Nhấn nút "🚀 Nhận diện"
    3. **Xem Kết Quả**: Tên thuốc, độ tin cậy, xác suất
    
    ### Train Model:
    1. Mở tab "🧠 Train model"
    2. Kiểm tra đường dẫn dataset
    3. Chọn classifier và feature extractor
    4. Nhấn "🚀 Bắt đầu train"
    
    ### Lưu Ý:
    - Ảnh sáng, rõ ràng sẽ cho kết quả tốt hơn
    - Nên chụp từ nhiều góc độ
    - Tránh ảnh mơ hồ, bị che khuất
    """)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: gray; font-size: 12px;"><p>Pill Recognition System v1.0.0</p></div>', unsafe_allow_html=True)