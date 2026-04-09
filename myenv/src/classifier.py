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
            so_vong: Giới hạn số vòng lặp train tối đa (map sang max_iter của SVM)
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
                random_state=42,   # Giúp kết quả ổn định hơn khi train lại
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

    def _kiem_tra_san_sang_du_doan(self):
        """Đảm bảo model/scaler đã sẵn sàng cho bước dự đoán."""
        if self.model is None:
            raise RuntimeError("Classifier chưa được khởi tạo.")

        if not self.da_train:
            raise RuntimeError("Model chưa được train. Hãy gọi train() trước.")

        if self.cac_lop is None or len(self.cac_lop) == 0:
            raise RuntimeError("Model chưa có danh sách class hợp lệ.")

        if not hasattr(self.model, "predict") or not hasattr(self.model, "predict_proba"):
            raise RuntimeError("Model hiện tại không hỗ trợ predict/predict_proba.")

        if not hasattr(self.scaler, "transform"):
            raise RuntimeError("Scaler hiện tại không hợp lệ.")

        if not hasattr(self.scaler, "n_features_in_"):
            raise RuntimeError("Scaler chưa được fit hoặc dữ liệu scaler bị thiếu.")

    def _chuan_hoa_du_lieu_train(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Chuẩn hóa và kiểm tra dữ liệu train trước khi fit model."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X phải có shape (n_samples, feature_dim).")

        if X.shape[0] == 0:
            raise ValueError("X không được rỗng.")

        if X.shape[1] == 0:
            raise ValueError("Feature vector phải có ít nhất 1 chiều.")

        if y.size == 0:
            raise ValueError("y không được rỗng.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Số mẫu của X và y không khớp: {X.shape[0]} != {y.shape[0]}"
            )

        if len(np.unique(y)) < 2:
            raise ValueError("Cần ít nhất 2 lớp khác nhau để train classifier.")

        if not np.isfinite(X).all():
            raise ValueError("X chứa giá trị không hợp lệ (NaN hoặc vô cực).")

        return X, y

    def _kiem_tra_so_chieu_feature(self, so_chieu_nhan_vao: int):
        """Kiểm tra số chiều feature có khớp với scaler đã train hay không."""
        so_chieu_mong_doi = getattr(self.scaler, "n_features_in_", None)

        if so_chieu_mong_doi is not None and so_chieu_nhan_vao != so_chieu_mong_doi:
            raise ValueError(
                "Số chiều feature không khớp với model đã train: "
                f"mong đợi {so_chieu_mong_doi}, nhận {so_chieu_nhan_vao}."
            )

    def _chuan_hoa_ma_tran_du_doan(self, X: np.ndarray) -> np.ndarray:
        """Chuẩn hóa input cho hàm dự đoán nhiều mẫu."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X phải có shape (n_samples, feature_dim).")

        if X.shape[0] == 0:
            raise ValueError("X không được rỗng.")

        if X.shape[1] == 0:
            raise ValueError("Feature vector phải có ít nhất 1 chiều.")

        if not np.isfinite(X).all():
            raise ValueError("X chứa giá trị không hợp lệ (NaN hoặc vô cực).")

        self._kiem_tra_so_chieu_feature(X.shape[1])
        return X

    def _chuan_hoa_feature_vector(self, feature_vector: np.ndarray) -> np.ndarray:
        """Chuẩn hóa input cho hàm dự đoán một ảnh."""
        feature_vector = np.asarray(feature_vector, dtype=np.float32)

        if feature_vector.ndim == 1:
            if feature_vector.size == 0:
                raise ValueError("feature_vector không được rỗng.")
            X = feature_vector.reshape(1, -1)
        elif feature_vector.ndim == 2 and feature_vector.shape[0] == 1:
            X = feature_vector
        else:
            raise ValueError(
                "feature_vector phải có shape (feature_dim,) hoặc (1, feature_dim)."
            )

        if X.shape[1] == 0:
            raise ValueError("Feature vector phải có ít nhất 1 chiều.")

        if not np.isfinite(X).all():
            raise ValueError("feature_vector chứa giá trị không hợp lệ (NaN hoặc vô cực).")

        self._kiem_tra_so_chieu_feature(X.shape[1])
        return X

    def _du_doan_noi_bo(self, X: np.ndarray) -> tuple:
        """Thực hiện dự đoán sau khi input đã được kiểm tra hợp lệ."""
        self._kiem_tra_san_sang_du_doan()

        # Normalize features bằng scaler đã fit ở bước train
        X_chuan_hoa = self.scaler.transform(X)

        # Dự đoán nhãn và xác suất
        nhan_du_doan = self.model.predict(X_chuan_hoa)
        xac_suat = self.model.predict_proba(X_chuan_hoa)

        if len(nhan_du_doan) != len(xac_suat):
            raise RuntimeError("Kết quả predict và predict_proba không đồng nhất.")

        if xac_suat.ndim != 2 or xac_suat.shape[1] != len(self.cac_lop):
            raise RuntimeError("Số cột xác suất không khớp với số class của model.")

        return nhan_du_doan, xac_suat

    def _lay_do_tin_cay_theo_nhan(self, nhan_du_doan, xac_suat: np.ndarray) -> float:
        """Lấy đúng xác suất ứng với nhãn mà model đã dự đoán."""
        chi_so_nhan = np.where(self.cac_lop == nhan_du_doan)[0]

        if len(chi_so_nhan) == 0:
            raise RuntimeError(f"Không tìm thấy nhãn dự đoán trong danh sách class: {nhan_du_doan}")

        return float(xac_suat[int(chi_so_nhan[0])])

    def _kiem_tra_du_lieu_model_da_tai(self, du_lieu) -> tuple:
        """Kiểm tra dữ liệu model đọc từ file có đầy đủ và hợp lệ hay không."""
        if not isinstance(du_lieu, dict):
            raise TypeError("File model không đúng định dạng dữ liệu mong đợi.")

        model = du_lieu.get("classifier")
        scaler = du_lieu.get("scaler")
        classes_raw = du_lieu.get("classes")

        if model is None:
            raise ValueError("File model không chứa classifier hợp lệ.")

        if scaler is None:
            raise ValueError("File model không chứa scaler hợp lệ.")

        if classes_raw is None:
            raise ValueError("File model không chứa danh sách class hợp lệ.")

        cac_lop = np.asarray(classes_raw)

        if cac_lop.ndim != 1 or cac_lop.size == 0:
            raise ValueError("Danh sách class trong file model không hợp lệ.")

        if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
            raise TypeError("Classifier trong file không hỗ trợ predict/predict_proba.")

        if not hasattr(scaler, "transform"):
            raise TypeError("Scaler trong file không hỗ trợ transform().")

        return model, scaler, cac_lop

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train mô hình classifier.
        
        Args:
            X: Feature vectors shape (n_samples, feature_dim)
            y: Labels shape (n_samples,)
        """
        try:
            X, y = self._chuan_hoa_du_lieu_train(X, y)
            ghi_log(f"Bắt đầu train | Số mẫu: {X.shape[0]} | Số chiều: {X.shape[1]}")

            # Bước 1: Normalize features (đưa về mean=0, std=1)
            X_chuan_hoa = self.scaler.fit_transform(X)

            # Bước 2: Train model
            self.model.fit(X_chuan_hoa, y)

            # Bước 3: Lưu danh sách các lớp
            self.cac_lop = np.asarray(self.model.classes_)

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
            X = self._chuan_hoa_ma_tran_du_doan(X)
            nhan_du_doan, xac_suat = self._du_doan_noi_bo(X)

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
            X = self._chuan_hoa_feature_vector(feature_vector)
            nhan_du_doan, tat_ca_xac_suat = self._du_doan_noi_bo(X)

            nhan = nhan_du_doan[0]
            xac_suat = tat_ca_xac_suat[0]

            # Lấy đúng xác suất ứng với nhãn mà model đã chọn
            do_tin_cay = self._lay_do_tin_cay_theo_nhan(nhan, xac_suat)

            # Tạo dict class → probability
            dict_xac_suat = {
                str(lop): float(xac_suat[i])
                for i, lop in enumerate(self.cac_lop)
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
                # Giữ cả 2 key để tương thích với dữ liệu đã lưu trước đây
                "max_iterations": self.so_vong,
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
                raise FileNotFoundError(f"Không tìm thấy file model: {duong_dan}")

            du_lieu = joblib.load(duong_dan)

            model, scaler, cac_lop = self._kiem_tra_du_lieu_model_da_tai(du_lieu)
            loai_classifier = str(du_lieu.get("classifier_type") or self.loai).lower()
            so_vong = du_lieu.get("max_iterations", du_lieu.get("training_epochs"))

            self.model = model
            self.scaler = scaler
            self.cac_lop = cac_lop
            self.loai = loai_classifier
            self.so_vong = so_vong
            self.da_train = True

            ghi_log(f"Đã tải model từ: {duong_dan} | Số lớp: {len(self.cac_lop)}")

        except Exception as e:
            ghi_loi(f"Không thể tải model: {str(e)}", e)
            raise

    def lay_cac_lop(self) -> np.ndarray:
        """Trả về danh sách các class labels."""
        return self.cac_lop if self.cac_lop is not None else np.array([])