"""
File này phân loại viên thuốc sử dụng Machine Learning.

Mục đích:
    - Nhận feature vector từ Feature Extractor
    - Phân loại viên thuốc thành một trong các lớp đã học
    - Trả về tên viên thuốc + độ tin cậy

Tại sao dùng ML thay vì DL?
    - Feature Extractor (DL) đã trích xuất đặc trưng tốt
    - ML Classifier (SVM, Random Forest) nhanh và hiệu quả
    - Dễ train với ít dữ liệu hơn

Kiến trúc:
    Ảnh → Feature Extractor (DL) → Feature Vector → Classifier (ML) → Kết quả
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .logger import ghi_log, ghi_loi


class PillClassifier:
    """
    Class phân loại viên thuốc dựa trên feature vector.
    
    Hỗ trợ 2 loại classifier:
        - SVM: Chính xác, phù hợp với dữ liệu ít
        - Random Forest: Nhanh, ít bị overfitting
    """
    def __init__(self, loai: str = "svm", so_vong: int = None):
        """
        Khởi tạo Classifier.
        
        Args:
            loai: Loại classifier ("svm" hoặc "random_forest")
            so_vong: Số vòng lặp train (chỉ áp dụng cho SVM)
        """
        self.loai = loai.lower()
        self.so_vong = so_vong
        self.model = None
        self.scaler = StandardScaler()  # Dùng để normalize features
        self.cac_lop = None  # Danh sách các lớp
        self.da_train = False

        ghi_log(f"Khởi tạo PillClassifier | Loại: {loai} | Số vòng: {so_vong or 'mặc định'}")

        try:
            self._tao_model()
            ghi_log("PillClassifier đã khởi tạo thành công")
        except Exception as e:
            ghi_loi(f"Không thể khởi tạo PillClassifier: {str(e)}", e)
            raise

    def _tao_model(self):
        """Tạo mô hình classifier (chưa train)."""
        if self.loai == "svm":
            # SVM với kernel RBF
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,  # Cho phép lấy xác suất
                max_iter=self.so_vong if self.so_vong is not None else -1,
                verbose=0,
            )
        elif self.loai == "random_forest":
            # Random Forest với 100 cây
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                verbose=0,
            )
        else:
            raise ValueError(f"Không hỗ trợ classifier: {self.loai}")

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train mô hình classifier.
        
        Args:
            X: Feature vectors shape (n_samples, feature_dim)
            y: Labels shape (n_samples,)
        """
        try:
            ghi_log(f"Bắt đầu train | Số mẫu: {X.shape[0]} | Số chiều: {X.shape[1]}")

            # Bước 1: Normalize features (đưa về mean=0, std=1)
            X_chuan_hoa = self.scaler.fit_transform(X)

            # Bước 2: Train model
            self.model.fit(X_chuan_hoa, y)

            # Bước 3: Lưu danh sách các lớp
            self.cac_lop = self.model.classes_

            self.da_train = True
            ghi_log(f"Train thành công | Số lớp: {len(self.cac_lop)}")

        except Exception as e:
            ghi_loi(f"Không thể train: {str(e)}", e)
            raise

    def du_doan(self, X: np.ndarray) -> tuple:
        """
        Dự đoán nhãn cho nhiều feature vectors.
        
        Args:
            X: Feature vectors shape (n_samples, feature_dim)
        
        Returns:
            Tuple (predicted_labels, probabilities)
        """
        try:
            if not self.da_train:
                raise RuntimeError("Model chưa được train. Hãy gọi train() trước.")

            # Normalize features
            X_chuan_hoa = self.scaler.transform(X)

            # Dự đoán nhãn
            nhan_du_doan = self.model.predict(X_chuan_hoa)

            # Lấy xác suất
            xac_suat = self.model.predict_proba(X_chuan_hoa)

            ghi_log(f"Dự đoán hoàn tất | Số mẫu: {X.shape[0]}")
            return nhan_du_doan, xac_suat

        except Exception as e:
            ghi_loi(f"Dự đoán thất bại: {str(e)}", e)
            raise

    def du_doan_mot_anh(self, feature_vector: np.ndarray) -> dict:
        """
        Dự đoán cho một ảnh duy nhất.
        
        Args:
            feature_vector: Feature vector shape (feature_dim,)
        
        Returns:
            Dict chứa:
                - predicted_label: Nhãn dự đoán
                - confidence: Độ tin cậy (0-1)
                - probabilities: Dict {class_label: probability}
        """
        try:
            if not self.da_train:
                raise RuntimeError("Model chưa được train. Hãy gọi train() trước.")

            # Reshape thành 2D array (1, feature_dim)
            X = feature_vector.reshape(1, -1)

            # Normalize
            X_chuan_hoa = self.scaler.transform(X)

            # Dự đoán
            nhan = self.model.predict(X_chuan_hoa)[0]
            xac_suat = self.model.predict_proba(X_chuan_hoa)[0]

            # Tìm xác suất cao nhất
            chi_so_cao = np.argmax(xac_suat)
            do_tin_cay = xac_suat[chi_so_cao]

            # Tạo dict class → probability
            dict_xac_suat = {
                str(self.cac_lop[i]): float(xac_suat[i])
                for i in range(len(self.cac_lop))
            }

            ket_qua = {
                "predicted_label": str(nhan),
                "confidence": float(do_tin_cay),
                "probabilities": dict_xac_suat,
            }

            ghi_log(f"Dự đoán: {nhan} (độ tin cậy: {do_tin_cay:.4f})")
            return ket_qua

        except Exception as e:
            ghi_loi(f"Dự đoán một ảnh thất bại: {str(e)}", e)
            raise

    def luu_model(self, duong_dan: str):
        """
        Lưu mô hình ra file.
        
        Args:
            duong_dan: Đường dẫn file output (.pkl)
        """
        try:
            if not self.da_train:
                raise RuntimeError("Không thể lưu model chưa train.")

            Path(duong_dan).parent.mkdir(parents=True, exist_ok=True)

            # Lưu tất cả thông tin cần thiết
            du_lieu = {
                "classifier": self.model,
                "scaler": self.scaler,
                "classes": self.cac_lop,
                "classifier_type": self.loai,
                "training_epochs": self.so_vong,
            }

            joblib.dump(du_lieu, duong_dan)
            ghi_log(f"Đã lưu model vào: {duong_dan}")

        except Exception as e:
            ghi_loi(f"Không thể lưu model: {str(e)}", e)
            raise

    def tai_model(self, duong_dan: str):
        """
        Tải mô hình đã train từ file.
        
        Args:
            duong_dan: Đường dẫn file model (.pkl)
        """
        try:
            if not Path(duong_dan).exists():
                ghi_log(f"Không tìm thấy file model: {duong_dan}")
                return

            du_lieu = joblib.load(duong_dan)

            self.model = du_lieu["classifier"]
            self.scaler = du_lieu["scaler"]
            self.cac_lop = du_lieu["classes"]
            self.loai = du_lieu["classifier_type"]
            self.so_vong = du_lieu.get("training_epochs")
            self.da_train = True

            ghi_log(f"Đã tải model từ: {duong_dan} | Số lớp: {len(self.cac_lop)}")

        except Exception as e:
            ghi_loi(f"Không thể tải model: {str(e)}", e)
            raise

    def lay_cac_lop(self) -> np.ndarray:
        """Trả về danh sách các class labels."""
        return self.cac_lop if self.cac_lop is not None else np.array([])