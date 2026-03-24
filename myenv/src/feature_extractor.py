"""
File này trích xuất đặc trưng từ ảnh sử dụng Deep Learning.

Mục đích:
    - Sử dụng mô hình AI đã train sẵn để trích xuất đặc trưng
    - Đặc trưng là một vector số (ví dụ: 1280 số) đại diện cho ảnh
    - Vector này sẽ được đưa vào ML Classifier để phân loại

Tại sao dùng pre-trained model?
    - Training từ đầu cần rất nhiều dữ liệu và thời gian
    - Pre-trained model đã học được cách nhận diện hình ảnh
    - Chúng ta chỉ cần "mượn" khả năng trích xuất đặc trưng

Kiến trúc:
    Ảnh → Feature Extractor (DL) → Feature Vector → Classifier (ML) → Kết quả
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import numpy as np

from .logger import ghi_log, ghi_loi
from .utils import anh_sang_tensor, tensor_sang_numpy, doc_anh_tu_bytes, doc_anh_tu_file


class FeatureExtractor:
    """
    Class trích xuất đặc trưng từ ảnh.
    
    Hỗ trợ 2 mô hình:
        - MobileNetV2: Nhẹ, nhanh
        - ResNet18: Chính xác hơn một chút
    """

    def __init__(self, ten_model: str = "mobilenetv2", thiet_bi: str = None):
        """
        Khởi tạo Feature Extractor.
        
        Args:
            ten_model: Tên mô hình ("mobilenetv2" hoặc "resnet18")
            thiet_bi: "cuda" (GPU) hoặc "cpu". Nếu None, tự động chọn
        """
        # Tự động chọn thiết bị (ưu tiên GPU nếu có)
        if thiet_bi is None:
            self.thiet_bi = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.thiet_bi = thiet_bi

        self.ten_model = ten_model.lower()
        self.model = None
        self.so_chieu = None  # Số chiều của feature vector

        ghi_log(f"Khởi tạo Feature Extractor | Model: {ten_model} | Thiết bị: {self.thiet_bi}")

        try:
            self._tai_model()
            ghi_log(f"Feature Extractor đã sẵn sàng | Số chiều: {self.so_chieu}")
        except Exception as e:
            ghi_loi(f"Không thể tải Feature Extractor: {str(e)}", e)
            raise

    def _tai_model(self):
        """
        Tải mô hình pre-trained từ torchvision.
        Cắt bỏ lớp phân loại cuối cùng để lấy feature extractor.
        """
        if self.ten_model == "mobilenetv2":
            # Tải MobileNetV2 đã train trên ImageNet
            model_goc = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
            
            # Cắt bỏ lớp classifier cuối cùng
            self.model = nn.Sequential(*list(model_goc.children())[:-1])
            
            # MobileNetV2 output: 1280 chiều
            self.so_chieu = 1280
            
        elif self.ten_model == "resnet18":
            # Tải ResNet18 đã train trên ImageNet
            model_goc = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            
            # Cắt bỏ adaptive avgpool và fc layer
            self.model = nn.Sequential(*list(model_goc.children())[:-2])
            
            # ResNet18 output: 512 chiều
            self.so_chieu = 512
            
        else:
            raise ValueError(f"Không hỗ trợ model: {self.ten_model}")

        # Đưa model lên thiết bị (GPU/CPU)
        self.model = self.model.to(self.thiet_bi)

        # Chuyển model sang chế độ evaluation
        self.model.eval()

    def trich_xuat_feature(self, tensor_anh: torch.Tensor) -> np.ndarray:
        """
        Trích xuất feature vector từ tensor ảnh.
        
        Args:
            tensor_anh: Tensor ảnh shape (1, 3, 224, 224)
        
        Returns:
            Feature vector dạng NumPy array
        """
        try:
            # Đưa tensor lên cùng thiết bị với model
            tensor_anh = tensor_anh.to(self.thiet_bi)

            # Trích xuất feature (không cần tính gradient)
            with torch.no_grad():
                features = self.model(tensor_anh)

            # Flatten: (1, C, H, W) → (1, C*H*W)
            features = features.view(features.size(0), -1)

            # Chuyển thành NumPy array
            feature_array = tensor_sang_numpy(features[0])

            ghi_log(f"Đã trích xuất feature | Shape: {feature_array.shape}")
            return feature_array

        except Exception as e:
            ghi_loi(f"Lỗi khi trích xuất feature: {str(e)}", e)
            raise

    def trich_xuat_tu_anh(self, nguon_anh) -> np.ndarray:
        """
        Trích xuất feature từ ảnh (file path hoặc bytes).
        
        Args:
            nguon_anh: Đường dẫn file ảnh (str) hoặc bytes của ảnh
        
        Returns:
            Feature vector dạng NumPy array
        """
        try:
            # Đọc ảnh dựa vào loại input
            if isinstance(nguon_anh, bytes):
                anh = doc_anh_tu_bytes(nguon_anh)
            else:
                anh = doc_anh_tu_file(nguon_anh)

            # Chuyển ảnh thành tensor
            tensor_anh = anh_sang_tensor(anh)

            # Trích xuất features
            features = self.trich_xuat_feature(tensor_anh)

            return features

        except Exception as e:
            ghi_loi(f"Lỗi khi trích xuất feature từ ảnh: {str(e)}", e)
            raise

    def luu_weights(self, duong_dan: str):
        """
        Lưu weights của mô hình ra file.
        
        Args:
            duong_dan: Đường dẫn file output (.pth)
        """
        try:
            Path(duong_dan).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), duong_dan)
            ghi_log(f"Đã lưu weights vào: {duong_dan}")
        except Exception as e:
            ghi_loi(f"Không thể lưu weights: {str(e)}", e)
            raise

    def tai_weights(self, duong_dan: str):
        """
        Tải weights từ file.
        
        Args:
            duong_dan: Đường dẫn file weights (.pth)
        """
        try:
            if not Path(duong_dan).exists():
                ghi_log(f"Không tìm thấy file weights: {duong_dan}. Sử dụng weights pre-trained.")
                return

            self.model.load_state_dict(torch.load(duong_dan, map_location=self.thiet_bi))
            self.model.eval()
            ghi_log(f"Đã tải weights từ: {duong_dan}")
        except Exception as e:
            ghi_loi(f"Không thể tải weights: {str(e)}", e)
            raise

    def lay_so_chieu(self) -> int:
        """
        Trả về số chiều của feature vector.
        
        Returns:
            Số chiều (VD: 1280 cho MobileNetV2, 512 cho ResNet18)
        """
        return self.so_chieu