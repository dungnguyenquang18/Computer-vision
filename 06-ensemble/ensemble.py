"""
Module 6: Ensemble Model for combining 3D CNN and 3D U-Net predictions
Kết hợp kết quả từ cả 3D CNN và 3D U-Net để đưa ra quyết định phân đoạn cuối cùng
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict


class EnsembleModel:
    """
    Ensemble model để kết hợp predictions từ 3D CNN và 3D U-Net
    
    Hỗ trợ 3 strategies:
    - weighted: Trung bình có trọng số (weighted averaging)
    - voting: Bỏ phiếu đa số (majority voting)
    - hybrid: Kết hợp cả hai (weighted cho low confidence, voting cho high confidence)
    """
    
    def __init__(
        self, 
        alpha: float = 0.4, 
        beta: float = 0.6, 
        strategy: str = 'weighted',
        confidence_threshold: float = 0.8,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Khởi tạo Ensemble Model
        
        Args:
            alpha: Trọng số cho CNN (thường 0.4)
            beta: Trọng số cho U-Net (thường 0.6, U-Net tốt hơn ở biên)
            strategy: Chiến lược ensemble ('weighted', 'voting', 'hybrid')
            confidence_threshold: Ngưỡng confidence cho hybrid strategy (0.8)
            device: Device để chạy ('cuda' hoặc 'cpu')
        """
        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta phải bằng 1"
        assert strategy in ['weighted', 'voting', 'hybrid'], \
            "strategy phải là 'weighted', 'voting', hoặc 'hybrid'"
        
        self.alpha = alpha
        self.beta = beta
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.device = device
        
    def reconstruct_from_patches(
        self, 
        patches: torch.Tensor, 
        positions: List[Tuple[int, int, int]],
        volume_shape: Tuple[int, int, int, int],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: Tuple[int, int, int] = (64, 64, 64)
    ) -> torch.Tensor:
        """
        Ghép các patches thành volume hoàn chỉnh
        Xử lý overlapping regions bằng averaging
        
        Args:
            patches: Tensor chứa các patches [N, C, D, H, W]
            positions: List các vị trí bắt đầu của mỗi patch [(z, y, x), ...]
            volume_shape: Kích thước volume output [C, D, H, W]
            patch_size: Kích thước mỗi patch (D, H, W)
            overlap: Mức độ overlap giữa các patches (D, H, W)
            
        Returns:
            volume: Volume đã được reconstruct [C, D, H, W]
        """
        C, D, H, W = volume_shape
        volume = torch.zeros((C, D, H, W), dtype=torch.float32, device=self.device)
        counts = torch.zeros((D, H, W), dtype=torch.float32, device=self.device)
        
        # Ghép từng patch vào volume
        for i, (z_start, y_start, x_start) in enumerate(positions):
            patch = patches[i]  # [C, patch_d, patch_h, patch_w]
            patch_d, patch_h, patch_w = patch_size
            
            z_end = min(z_start + patch_d, D)
            y_end = min(y_start + patch_h, H)
            x_end = min(x_start + patch_w, W)
            
            # Tính kích thước thực tế của patch (có thể nhỏ hơn patch_size ở biên)
            actual_d = z_end - z_start
            actual_h = y_end - y_start
            actual_w = x_end - x_start
            
            # Cộng dồn patch vào volume
            volume[:, z_start:z_end, y_start:y_end, x_start:x_end] += \
                patch[:, :actual_d, :actual_h, :actual_w]
            
            # Đếm số lần mỗi voxel được cộng
            counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1
        
        # Chia trung bình cho các vùng overlap
        counts = torch.clamp(counts, min=1)  # Tránh chia cho 0
        volume = volume / counts.unsqueeze(0)
        
        return volume
    
    def weighted_average(
        self, 
        prob_cnn: torch.Tensor, 
        prob_unet: torch.Tensor
    ) -> torch.Tensor:
        """
        Trung bình có trọng số
        P_final = α · P_cnn + β · P_unet
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            
        Returns:
            prob_final: Probability map kết hợp [C, D, H, W]
        """
        return self.alpha * prob_cnn + self.beta * prob_unet
    
    def majority_voting(
        self, 
        prob_cnn: torch.Tensor, 
        prob_unet: torch.Tensor
    ) -> torch.Tensor:
        """
        Bỏ phiếu đa số (Majority Voting)
        - Lấy class có xác suất cao nhất từ mỗi model
        - Nếu trùng nhau → chọn class đó
        - Nếu khác nhau → chọn theo model có confidence cao hơn
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            
        Returns:
            prob_final: Probability map kết hợp [C, D, H, W]
        """
        # Lấy class dự đoán và confidence từ mỗi model
        conf_cnn, pred_cnn = torch.max(prob_cnn, dim=0)  # [D, H, W]
        conf_unet, pred_unet = torch.max(prob_unet, dim=0)  # [D, H, W]
        
        # Tạo output probability map
        C, D, H, W = prob_cnn.shape
        prob_final = torch.zeros_like(prob_cnn)
        
        # Case 1: Cả hai model đồng ý → lấy trung bình confidence
        agree_mask = (pred_cnn == pred_unet)
        for c in range(C):
            class_mask = (pred_cnn == c) & agree_mask
            prob_final[c][class_mask] = (prob_cnn[c][class_mask] + prob_unet[c][class_mask]) / 2
        
        # Case 2: Hai model không đồng ý → chọn model có confidence cao hơn
        disagree_mask = ~agree_mask
        cnn_better = (conf_cnn > conf_unet) & disagree_mask
        unet_better = (conf_unet >= conf_cnn) & disagree_mask
        
        for c in range(C):
            # Nơi CNN tốt hơn
            prob_final[c][cnn_better] = prob_cnn[c][cnn_better]
            # Nơi U-Net tốt hơn
            prob_final[c][unet_better] = prob_unet[c][unet_better]
        
        return prob_final
    
    def hybrid_approach(
        self, 
        prob_cnn: torch.Tensor, 
        prob_unet: torch.Tensor
    ) -> torch.Tensor:
        """
        Hybrid Approach: Kết hợp cả weighted và voting
        - Weighted averaging cho vùng low confidence
        - Majority voting cho vùng high confidence
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            
        Returns:
            prob_final: Probability map kết hợp [C, D, H, W]
        """
        # Tính confidence của mỗi model
        conf_cnn, _ = torch.max(prob_cnn, dim=0)  # [D, H, W]
        conf_unet, _ = torch.max(prob_unet, dim=0)  # [D, H, W]
        
        # Confidence trung bình
        avg_conf = (conf_cnn + conf_unet) / 2
        
        # Mask cho high/low confidence
        high_conf_mask = avg_conf >= self.confidence_threshold
        low_conf_mask = ~high_conf_mask
        
        # Áp dụng weighted averaging cho low confidence
        prob_weighted = self.weighted_average(prob_cnn, prob_unet)
        
        # Áp dụng majority voting cho high confidence
        prob_voting = self.majority_voting(prob_cnn, prob_unet)
        
        # Kết hợp
        prob_final = torch.zeros_like(prob_cnn)
        prob_final[:, low_conf_mask] = prob_weighted[:, low_conf_mask]
        prob_final[:, high_conf_mask] = prob_voting[:, high_conf_mask]
        
        return prob_final
    
    def ensemble(
        self, 
        prob_cnn: torch.Tensor, 
        prob_unet: torch.Tensor
    ) -> torch.Tensor:
        """
        Áp dụng chiến lược ensemble đã chọn
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            
        Returns:
            prob_final: Probability map kết hợp [C, D, H, W]
        """
        # Chuyển sang device nếu cần
        if prob_cnn.device != self.device:
            prob_cnn = prob_cnn.to(self.device)
        if prob_unet.device != self.device:
            prob_unet = prob_unet.to(self.device)
        
        if self.strategy == 'weighted':
            return self.weighted_average(prob_cnn, prob_unet)
        elif self.strategy == 'voting':
            return self.majority_voting(prob_cnn, prob_unet)
        elif self.strategy == 'hybrid':
            return self.hybrid_approach(prob_cnn, prob_unet)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def argmax_to_mask(self, prob: torch.Tensor) -> torch.Tensor:
        """
        Chuyển probability map thành segmentation mask
        Tại mỗi voxel, chọn lớp có xác suất cao nhất
        
        Args:
            prob: Probability map [C, D, H, W]
            
        Returns:
            mask: Segmentation mask [D, H, W]
        """
        # Lấy index của class có xác suất cao nhất
        mask = torch.argmax(prob, dim=0)  # [D, H, W]
        return mask
    
    def predict(
        self,
        prob_cnn: torch.Tensor,
        prob_unet: torch.Tensor,
        return_probabilities: bool = False
    ) -> torch.Tensor:
        """
        Pipeline đầy đủ: ensemble + argmax
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            return_probabilities: Có trả về probability map không (default: False)
            
        Returns:
            mask hoặc (mask, prob) nếu return_probabilities=True
        """
        # Ensemble
        prob_final = self.ensemble(prob_cnn, prob_unet)
        
        # Argmax
        mask = self.argmax_to_mask(prob_final)
        
        if return_probabilities:
            return mask, prob_final
        return mask
    
    def get_statistics(
        self,
        prob_cnn: torch.Tensor,
        prob_unet: torch.Tensor
    ) -> Dict[str, float]:
        """
        Tính toán thống kê về sự đồng thuận giữa 2 models
        
        Args:
            prob_cnn: Probability map từ CNN [C, D, H, W]
            prob_unet: Probability map từ U-Net [C, D, H, W]
            
        Returns:
            stats: Dictionary chứa các thống kê
        """
        pred_cnn = torch.argmax(prob_cnn, dim=0)
        pred_unet = torch.argmax(prob_unet, dim=0)
        
        # Agreement rate
        agreement = (pred_cnn == pred_unet).float().mean().item()
        
        # Average confidence
        conf_cnn = torch.max(prob_cnn, dim=0)[0].mean().item()
        conf_unet = torch.max(prob_unet, dim=0)[0].mean().item()
        
        return {
            'agreement_rate': agreement,
            'avg_confidence_cnn': conf_cnn,
            'avg_confidence_unet': conf_unet,
            'avg_confidence_overall': (conf_cnn + conf_unet) / 2
        }
