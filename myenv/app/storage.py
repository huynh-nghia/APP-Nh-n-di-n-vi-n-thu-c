"""
Module lưu trữ dữ liệu predictions cho analytics.

Mô tả:
    - Lưu thông tin predictions vào SQLite
    - Tự động migrate dữ liệu cũ từ predictions.json (nếu có)
    - Load lịch sử predictions
    - Xóa dữ liệu cũ (nếu cần)
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ========================================
# Cấu hình đường dẫn
# ========================================
DATA_DIR = Path("data")                    # Thư mục lưu dữ liệu
DB_FILE = DATA_DIR / "predictions.db"      # SQLite DB lưu lịch sử dự đoán
LEGACY_JSON_FILE = DATA_DIR / "predictions.json"  # File JSON cũ (nếu có)
MIGRATED_JSON_FILE = DATA_DIR / "predictions.json.migrated"


# ========================================
# Đảm bảo thư mục dữ liệu tồn tại
# ========================================
def ensure_data_directory():
    """Tạo thư mục data nếu chưa tồn tại."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Mở kết nối SQLite với row factory dạng dict-like."""
    ensure_data_directory()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    """Đảm bảo schema DB luôn đúng phiên bản hiện tại."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_pill TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_json TEXT NOT NULL,
            filename TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 1
        )
        """
    )

    # Tương thích ngược: thêm cột success nếu DB được tạo từ version cũ
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "success" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN success INTEGER NOT NULL DEFAULT 1")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_pill ON predictions(predicted_pill)"
    )


def _safe_float(value, default: float = 0.0) -> float:
    """Ép kiểu float an toàn."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_probabilities(probabilities) -> Dict[str, float]:
    """Chuẩn hóa dict probabilities về dạng {str: float}."""
    if not isinstance(probabilities, dict):
        return {}

    normalized = {}
    for k, v in probabilities.items():
        normalized[str(k)] = _safe_float(v)
    return normalized


def _migrate_legacy_json_if_needed(conn: sqlite3.Connection):
    """Tự động migrate data từ predictions.json sang SQLite nếu cần."""
    if not LEGACY_JSON_FILE.exists():
        return

    row_count = conn.execute("SELECT COUNT(*) AS total FROM predictions").fetchone()["total"]
    if row_count > 0:
        return

    try:
        with open(LEGACY_JSON_FILE, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)
    except Exception as e:
        print(f"Không thể đọc file legacy JSON: {e}")
        return

    if not isinstance(legacy_data, list) or not legacy_data:
        return

    rows_to_insert = []
    for item in legacy_data:
        if not isinstance(item, dict):
            continue

        timestamp = str(item.get("timestamp") or datetime.now().isoformat())
        predicted_pill = str(item.get("predicted_pill", "unknown"))
        confidence = _safe_float(item.get("confidence"), 0.0)
        probabilities = _normalize_probabilities(item.get("probabilities", {}))
        filename = str(item.get("filename", "unknown"))
        success = 1 if item.get("success", True) else 0

        rows_to_insert.append(
            (
                timestamp,
                predicted_pill,
                confidence,
                json.dumps(probabilities, ensure_ascii=False),
                filename,
                success,
            )
        )

    if not rows_to_insert:
        return

    conn.executemany(
        """
        INSERT INTO predictions (timestamp, predicted_pill, confidence, probabilities_json, filename, success)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )

    try:
        target = MIGRATED_JSON_FILE
        if target.exists():
            target = DATA_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.migrated"
        LEGACY_JSON_FILE.rename(target)
    except Exception as e:
        print(f"Migrate thành công nhưng không thể đổi tên file legacy: {e}")


def initialize_db():
    """Khởi tạo bảng predictions nếu chưa tồn tại."""
    with get_connection() as conn:
        _ensure_schema(conn)
        _migrate_legacy_json_if_needed(conn)


def row_to_prediction(row: sqlite3.Row) -> Dict:
    """Chuyển row SQLite sang dict tương thích UI hiện tại."""
    row_keys = row.keys()
    raw_success = row["success"] if "success" in row_keys else 1

    return {
        "timestamp": row["timestamp"],
        "predicted_pill": row["predicted_pill"],
        "confidence": float(row["confidence"]),
        "probabilities": json.loads(row["probabilities_json"]) if row["probabilities_json"] else {},
        "filename": row["filename"],
        "success": bool(raw_success),
    }


# ========================================
# Lưu kết quả dự đoán
# ========================================
def save_prediction(
    predicted_pill: str,
    confidence: float,
    probabilities: Dict[str, float],
    filename: str = None,
    success: bool = True,
) -> bool:
    """
    Lưu kết quả dự đoán vào SQLite.

    Args:
        predicted_pill: Tên viên thuốc được nhận diện
        confidence: Độ tin cậy (0.0 - 1.0)
        probabilities: Xác suất cho từng loại viên thuốc (dict)
        filename: Tên file ảnh (có thể để None)

    Returns:
        True nếu lưu thành công, False nếu thất bại
    """
    try:
        initialize_db()
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "predicted_pill": predicted_pill,
            "confidence": float(confidence),
            "probabilities": _normalize_probabilities(probabilities),
            "filename": filename or "unknown",
            "success": bool(success),
        }

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO predictions (timestamp, predicted_pill, confidence, probabilities_json, filename, success)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_record["timestamp"],
                    prediction_record["predicted_pill"],
                    prediction_record["confidence"],
                    json.dumps(prediction_record["probabilities"], ensure_ascii=False),
                    prediction_record["filename"],
                    1 if prediction_record["success"] else 0,
                ),
            )

        return True

    except Exception as e:
        print(f"Lỗi khi lưu prediction: {e}")
        return False


# ========================================
# Đọc toàn bộ lịch sử dự đoán
# ========================================
def load_all_predictions() -> List[Dict]:
    """
    Đọc toàn bộ lịch sử dự đoán từ SQLite.

    Returns:
        Danh sách các bản ghi dự đoán (mỗi bản ghi là một dict)
    """
    try:
        initialize_db()

        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, predicted_pill, confidence, probabilities_json, filename, success
                FROM predictions
                ORDER BY timestamp DESC
                """
            ).fetchall()

        return [row_to_prediction(row) for row in rows]

    except Exception as e:
        print(f"Lỗi khi load predictions: {e}")
        return []


# ========================================
# Xóa toàn bộ dữ liệu dự đoán
# ========================================
def clear_predictions() -> bool:
    """
    Xóa toàn bộ dữ liệu dự đoán.

    Cách hoạt động:
    1. Đảm bảo DB tồn tại
    2. Xóa toàn bộ records trong bảng predictions
    3. Trả về True nếu thành công

    Returns:
        True nếu xóa thành công, False nếu thất bại
    """
    try:
        initialize_db()
        with get_connection() as conn:
            conn.execute("DELETE FROM predictions")
        return True
    except Exception as e:
        print(f"Lỗi khi xóa predictions: {e}")
        return False


# ========================================
# Đếm số lượng dự đoán theo loại viên thuốc
# ========================================
def get_prediction_stats() -> Dict[str, int]:
    """
    Thống kê số lượng dự đoán theo từng loại viên thuốc.

    Cách hoạt động:
    1. Đọc toàn bộ lịch sử dự đoán
    2. Đếm số lần xuất hiện của mỗi loại viên thuốc
    3. Trả về dict với key là tên viên thuốc, value là số lần

    Returns:
        Dict thống kê dạng {"pill_name": count, ...}
    """
    try:
        initialize_db()
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT predicted_pill, COUNT(*) AS count
                FROM predictions
                GROUP BY predicted_pill
                """
            ).fetchall()

        return {row["predicted_pill"]: int(row["count"]) for row in rows}

    except Exception as e:
        print(f"Lỗi khi tính stats: {e}")
        return {}


# ========================================
# Lấy các dự đoán gần nhất
# ========================================
def get_recent_predictions(limit: int = 10) -> List[Dict]:
    """
    Lấy các dự đoán gần nhất.

    Args:
        limit: Số lượng dự đoán gần nhất cần lấy (mặc định 10)

    Returns:
        Danh sách các dự đoán gần nhất (được sắp xếp theo thời gian)
    """
    try:
        initialize_db()
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, predicted_pill, confidence, probabilities_json, filename, success
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(0, int(limit)),),
            ).fetchall()

        return [row_to_prediction(row) for row in rows]
    except Exception as e:
        print(f"Lỗi khi lấy recent predictions: {e}")
        return []


# ========================================
# Tìm dự đoán theo tên viên thuốc
# ========================================
def get_predictions_by_pill(pill_name: str) -> List[Dict]:
    """
    Tìm các dự đoán theo tên viên thuốc.

    Args:
        pill_name: Tên viên thuốc cần tìm

    Returns:
        Danh sách các dự đoán có tên viên thuốc khớp
    """
    try:
        initialize_db()
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, predicted_pill, confidence, probabilities_json, filename, success
                FROM predictions
                WHERE predicted_pill = ?
                ORDER BY timestamp DESC
                """,
                (pill_name,),
            ).fetchall()

        return [row_to_prediction(row) for row in rows]
    except Exception as e:
        print(f"Lỗi khi tìm predictions theo pill: {e}")
        return []


# ========================================
# Lấy độ tin cậy trung bình theo viên thuốc
# ========================================
def get_average_confidence_by_pill(pill_name: str) -> float:
    """
    Tính độ tin cậy trung bình cho một loại viên thuốc.

    Args:
        pill_name: Tên viên thuốc

    Returns:
        Độ tin cậy trung bình (0.0 - 1.0), hoặc 0.0 nếu không có dữ liệu
    """
    try:
        predictions = get_predictions_by_pill(pill_name)
        if not predictions:
            return 0.0

        total_confidence = sum(pred.get("confidence", 0.0) for pred in predictions)
        return total_confidence / len(predictions)

    except Exception as e:
        print(f"Lỗi khi tính average confidence: {e}")
        return 0.0


# ========================================
# In thống kê cơ bản
# ========================================
def print_stats():
    """
    In thống kê cơ bản ra console.

    Hiển thị:
    - Tổng số dự đoán
    - Thống kê theo loại viên thuốc
    - Độ tin cậy trung bình cho từng loại
    """
    try:
        predictions = load_all_predictions()
        stats = get_prediction_stats()

        print("\n=== THỐNG KÊ DỰ ĐOÁN ===")
        print(f"Tổng số dự đoán: {len(predictions)}")
        success_count = sum(1 for p in predictions if p.get("success", False))
        print(f"Số dự đoán thành công: {success_count}")
        print("\nThống kê theo loại viên thuốc:")

        for pill_name, count in stats.items():
            avg_conf = get_average_confidence_by_pill(pill_name)
            print(f"  {pill_name}: {count} lần (độ tin cậy TB: {avg_conf:.2f})")

        print("=" * 30)

    except Exception as e:
        print(f"Lỗi khi in stats: {e}")


# ========================================
# Ví dụ sử dụng
# ========================================
if __name__ == "__main__":
    initialize_db()

    # Ví dụ cách sử dụng các hàm

    # 1. Lưu một dự đoán mẫu
    save_prediction(
        predicted_pill="paracetamol",
        confidence=0.95,
        probabilities={"paracetamol": 0.95, "ibuprofen": 0.03, "aspirin": 0.02},
        filename="pill_001.jpg"
    )

    # 2. In thống kê
    print_stats()

    # 3. Lấy các dự đoán gần nhất
    recent = get_recent_predictions(5)
    print(f"\n5 dự đoán gần nhất: {len(recent)}")

    # 4. Lấy thống kê cho một loại viên thuốc cụ thể
    paracetamol_preds = get_predictions_by_pill("paracetamol")
    print(f"\nSố dự đoán paracetamol: {len(paracetamol_preds)}")