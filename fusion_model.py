import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn3d import CNN3DModel
from unet3d import UNet3D

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaBetaFusion(nn.Module):
    def __init__(self, model_cnn, model_unet, init_val=0.5, freeze_backbones=True):
        """
        Args:
            model_cnn: Mô hình 3D CNN đã định nghĩa
            model_unet: Mô hình 3D U-Net đã định nghĩa
            init_val: Giá trị khởi tạo cho alpha và beta (thường là 0.5)
            freeze_backbones: Có đóng băng trọng số của 2 model con hay không
        """
        super(AlphaBetaFusion, self).__init__()
        
        self.model_cnn = model_cnn
        self.model_unet = model_unet
        
        # --- ĐỊNH NGHĨA ALPHA VÀ BETA ---
        # nn.Parameter biến tensor thường thành tham số có thể train (có gradient)
        # requires_grad=True mặc định
        self.alpha = nn.Parameter(torch.tensor(init_val))
        self.beta = nn.Parameter(torch.tensor(init_val))
        
        # Tùy chọn: Đóng băng 2 model gốc để chỉ train alpha và beta (nhanh hơn)
        if freeze_backbones:
            for param in self.model_cnn.parameters():
                param.requires_grad = False
            for param in self.model_unet.parameters():
                param.requires_grad = False

    def forward(self, x):
        # 1. Forward qua từng model để lấy logits
        logits_cnn = self.model_cnn(x)
        logits_unet = self.model_unet(x)
        
        # 2. Xử lý chênh lệch kích thước (QUAN TRỌNG)
        # Theo tài liệu[cite: 75, 76], U-Net gốc thường có output nhỏ hơn input
        # do không dùng padding. Ta cần resize logits_unet về bằng logits_cnn.
        if logits_cnn.shape != logits_unet.shape:
            logits_unet = F.interpolate(
                logits_unet, 
                size=logits_cnn.shape[2:], # Lấy (D, H, W) của CNN làm chuẩn
                mode='trilinear', 
                align_corners=False
            )
            
        # 3. Nhân hệ số và cộng gộp (Fusion)
        # Công thức: Final = alpha * CNN + beta * UNet
        fused_logits = (self.alpha * logits_cnn) + (self.beta * logits_unet)
        
        return fused_logits

# --- Cách sử dụng ---
# model = AlphaBetaFusion(your_cnn, your_unet)
# print(f"Alpha ban đầu: {model.alpha.item()}, Beta ban đầu: {model.beta.item()}")

# Sau khi train xong, bạn có thể xem mô hình tin tưởng mạng nào hơn:
# print(f"Alpha sau train: {model.alpha.item()}, Beta sau train: {model.beta.item()}")