"""
Full Pipeline: Tích hợp toàn bộ luồng xử lý từ MRI thô đến kết quả phân đoạn cuối cùng
Pipeline: Input MRI → Preprocessing → GLCM → VPT → CNN & U-Net → Ensemble → Output Mask
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import các modules với absolute path
_parent_dir = Path(__file__).parent
sys.path.insert(0, str(_parent_dir / "preprocessing"))
sys.path.insert(0, str(_parent_dir / "glcm_extraction"))
sys.path.insert(0, str(_parent_dir / "vpt_selection"))
sys.path.insert(0, str(_parent_dir / "cnn3d"))
sys.path.insert(0, str(_parent_dir / "unet3d"))
sys.path.insert(0, str(_parent_dir / "ensemble"))

from preprocessor import Preprocessor
from glcm_extractor import GLCMExtractor
from vpt_selector import VPTSelector
from cnn3d_model import CNN3DModel
from unet3d_model import UNet3DModel
from ensemble import EnsembleModel


class BrainTumorSegmentationPipeline:
    """
    Pipeline đầy đủ cho phân đoạn u não tự động
    
    Các bước:
    1. Preprocessing: Zero-padding (155→160) + Z-score normalization
    2. GLCM Extraction: Trích xuất 20 features (4 channels × 5 Haralick features)
    3. VPT Selection: Lựa chọn N features quan trọng nhất
    4. 3D CNN: Phân loại dựa trên ngữ cảnh cục bộ
    5. 3D U-Net: Phân đoạn với skip connections
    6. Ensemble: Kết hợp CNN + U-Net để ra quyết định cuối
    """
    
    def __init__(self, 
                 target_shape=(240, 240, 160, 4),
                 glcm_window_size=3,
                 glcm_levels=16,
                 vpt_n_features=10,
                 vpt_method='variance',
                 patch_size=(128, 128, 128),
                 num_classes=4,
                 ensemble_strategy='weighted',
                 ensemble_alpha=0.4,
                 ensemble_beta=0.6,
                 device='auto'):
        """
        Khởi tạo pipeline
        
        Args:
            target_shape: Kích thước sau preprocessing (H, W, D, C)
            glcm_window_size: Kích thước cửa sổ GLCM (3, 5, 7, ...)
            glcm_levels: Số mức xám cho GLCM (8, 16, 32)
            vpt_n_features: Số features giữ lại sau VPT
            vpt_method: Phương pháp VPT ('variance', 'pca', 'correlation')
            patch_size: Kích thước patch cho CNN/U-Net
            num_classes: Số lớp phân đoạn (4: Background, Necrotic, Edema, Enhancing)
            ensemble_strategy: Chiến lược ensemble ('weighted', 'voting', 'hybrid')
            ensemble_alpha: Trọng số CNN
            ensemble_beta: Trọng số U-Net
            device: 'auto', 'cuda', hoặc 'cpu'
        """
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"\n{'='*80}")
        print(f"INITIALIZING BRAIN TUMOR SEGMENTATION PIPELINE")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        
        # Step 1: Preprocessing
        self.preprocessor = Preprocessor(target_shape=target_shape)
        
        # Step 2: GLCM Extraction
        self.glcm_extractor = GLCMExtractor(
            window_size=glcm_window_size,
            levels=glcm_levels
        )
        
        # Step 3: VPT Selection
        self.vpt_selector = VPTSelector(
            n_features=vpt_n_features,
            selection_method=vpt_method
        )
        
        # Step 4: 3D CNN Model
        # Input shape cho CNN/U-Net là patch_size + số features sau VPT
        cnn_input_shape = (*patch_size, vpt_n_features)
        self.cnn_model = CNN3DModel(
            input_shape=cnn_input_shape,
            num_classes=num_classes
        )
        
        # Step 5: 3D U-Net Model
        self.unet_model = UNet3DModel(
            input_shape=cnn_input_shape,
            num_classes=num_classes
        )
        
        # Step 6: Ensemble
        self.ensemble_model = EnsembleModel(
            alpha=ensemble_alpha,
            beta=ensemble_beta,
            strategy=ensemble_strategy,
            device=self.device
        )
        
        # Pipeline config
        self.target_shape = target_shape
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        print(f"\n{'='*80}")
        print(f"PIPELINE INITIALIZED SUCCESSFULLY")
        print(f"{'='*80}\n")
    
    def preprocess_step(self, volume):
        """
        Bước 1: Tiền xử lý
        Input: (H, W, D, C) - Raw MRI
        Output: (240, 240, 160, 4) - Preprocessed
        """
        print(f"\n{'='*80}")
        print(f"STEP 1: PREPROCESSING")
        print(f"{'='*80}")
        
        processed = self.preprocessor.preprocess(volume)
        
        print(f"✓ Preprocessing completed")
        print(f"  Input shape: {volume.shape}")
        print(f"  Output shape: {processed.shape}")
        
        return processed
    
    def extract_glcm_features_step(self, volume, fast_mode=False, stride=5):
        """
        Bước 2: Trích xuất đặc trưng GLCM
        Input: (240, 240, 160, 4) - Preprocessed
        Output: (240, 240, 160, 20) - GLCM features
        """
        print(f"\n{'='*80}")
        print(f"STEP 2: GLCM FEATURE EXTRACTION")
        print(f"{'='*80}")
        
        if fast_mode:
            print(f"[FAST MODE] Using stride={stride}")
            features = self.glcm_extractor.extract_features_fast(volume, stride=stride)
        else:
            features = self.glcm_extractor.extract_features(volume)
        
        print(f"✓ GLCM extraction completed")
        print(f"  Input shape: {volume.shape}")
        print(f"  Output shape: {features.shape}")
        
        return features
    
    def select_features_step(self, volume):
        """
        Bước 3: Lựa chọn đặc trưng VPT
        Input: (240, 240, 160, 20) - GLCM features
        Output: (240, 240, 160, N) - Selected features
        """
        print(f"\n{'='*80}")
        print(f"STEP 3: VPT FEATURE SELECTION")
        print(f"{'='*80}")
        
        selected = self.vpt_selector.select_features(volume)
        
        print(f"✓ Feature selection completed")
        print(f"  Input shape: {volume.shape}")
        print(f"  Output shape: {selected.shape}")
        
        return selected
    
    def predict_cnn_step(self, volume, stride=None):
        """
        Bước 4: Dự đoán với 3D CNN
        Input: (240, 240, 160, N) - Selected features
        Output: (240, 240, 160, num_classes) - CNN probabilities
        """
        print(f"\n{'='*80}")
        print(f"STEP 4: 3D CNN PREDICTION")
        print(f"{'='*80}")
        
        # Predict volume với patches
        prob_cnn = self.cnn_model.predict_volume(
            volume, 
            patch_size=self.patch_size,
            stride=stride
        )
        
        print(f"✓ CNN prediction completed")
        print(f"  Input shape: {volume.shape}")
        print(f"  Output shape: {prob_cnn.shape}")
        
        return prob_cnn
    
    def predict_unet_step(self, volume, stride=None):
        """
        Bước 5: Dự đoán với 3D U-Net
        Input: (240, 240, 160, N) - Selected features
        Output: (240, 240, 160, num_classes) - U-Net probabilities
        """
        print(f"\n{'='*80}")
        print(f"STEP 5: 3D U-NET PREDICTION")
        print(f"{'='*80}")
        
        # Predict volume với patches
        prob_unet = self.unet_model.predict_volume(
            volume,
            patch_size=self.patch_size,
            stride=stride
        )
        
        print(f"✓ U-Net prediction completed")
        print(f"  Input shape: {volume.shape}")
        print(f"  Output shape: {prob_unet.shape}")
        
        return prob_unet
    
    def ensemble_step(self, prob_cnn, prob_unet):
        """
        Bước 6: Ensemble predictions
        Input: 2 probability maps (240, 240, 160, num_classes)
        Output: Final mask (240, 240, 160)
        """
        print(f"\n{'='*80}")
        print(f"STEP 6: ENSEMBLE & POST-PROCESSING")
        print(f"{'='*80}")
        
        # Convert numpy to torch
        prob_cnn_torch = torch.FloatTensor(prob_cnn).permute(3, 0, 1, 2)  # (C, H, W, D)
        prob_unet_torch = torch.FloatTensor(prob_unet).permute(3, 0, 1, 2)
        
        # Ensemble
        mask = self.ensemble_model.predict(prob_cnn_torch, prob_unet_torch)
        
        # Convert back to numpy
        mask_np = mask.cpu().numpy()
        
        # Statistics
        stats = self.ensemble_model.get_statistics(prob_cnn_torch, prob_unet_torch)
        
        print(f"✓ Ensemble completed")
        print(f"  Strategy: {self.ensemble_model.strategy}")
        print(f"  Agreement rate: {stats['agreement_rate']:.4f}")
        print(f"  CNN confidence: {stats['avg_confidence_cnn']:.4f}")
        print(f"  U-Net confidence: {stats['avg_confidence_unet']:.4f}")
        print(f"  Output shape: {mask_np.shape}")
        
        return mask_np
    
    def postprocess_step(self, mask):
        """
        Bước 7: Post-processing (cắt padding)
        Input: (240, 240, 160)
        Output: (240, 240, 155) - Original depth
        """
        print(f"\n{'='*80}")
        print(f"STEP 7: POST-PROCESSING")
        print(f"{'='*80}")
        
        # Remove padding (160 → 155)
        mask_original = mask[:, :, :155]
        
        print(f"✓ Post-processing completed")
        print(f"  Input shape: {mask.shape}")
        print(f"  Output shape: {mask_original.shape}")
        
        return mask_original
    
    def predict(self, volume, fast_mode=True, glcm_stride=10, pred_stride=None):
        """
        Pipeline đầy đủ: từ MRI thô đến mask cuối cùng
        
        Args:
            volume: Raw MRI volume (H, W, D, C)
            fast_mode: Dùng fast mode cho GLCM (default: True)
            glcm_stride: Stride cho GLCM fast mode (default: 10)
            pred_stride: Stride cho CNN/U-Net prediction (default: None = no overlap)
            
        Returns:
            mask: Final segmentation mask (240, 240, 155)
        """
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        print(f"  STARTING FULL BRAIN TUMOR SEGMENTATION PIPELINE")
        print(f"{'#'*80}")
        print(f"{'#'*80}\n")
        
        # Step 1: Preprocessing
        preprocessed = self.preprocess_step(volume)
        
        # Step 2: GLCM Feature Extraction
        glcm_features = self.extract_glcm_features_step(
            preprocessed, 
            fast_mode=fast_mode, 
            stride=glcm_stride
        )
        
        # Step 3: VPT Feature Selection
        selected_features = self.select_features_step(glcm_features)
        
        # Step 4: CNN Prediction
        prob_cnn = self.predict_cnn_step(selected_features, stride=pred_stride)
        
        # Step 5: U-Net Prediction
        prob_unet = self.predict_unet_step(selected_features, stride=pred_stride)
        
        # Step 6: Ensemble
        final_mask = self.ensemble_step(prob_cnn, prob_unet)
        
        # Step 7: Post-processing
        final_mask = self.postprocess_step(final_mask)
        
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        print(f"  PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"  Final mask shape: {final_mask.shape}")
        print(f"  Unique labels: {np.unique(final_mask)}")
        print(f"{'#'*80}")
        print(f"{'#'*80}\n")
        
        return final_mask
    
    def get_pipeline_summary(self):
        """In ra tóm tắt pipeline"""
        print(f"\n{'='*80}")
        print(f"PIPELINE SUMMARY")
        print(f"{'='*80}")
        print(f"1. Preprocessing:")
        print(f"   - Target shape: {self.target_shape}")
        print(f"   - Operations: Zero-padding + Z-score normalization")
        print(f"\n2. GLCM Extraction:")
        print(f"   - Window size: {self.glcm_extractor.window_size}")
        print(f"   - Gray levels: {self.glcm_extractor.levels}")
        print(f"   - Output: 20 feature channels")
        print(f"\n3. VPT Selection:")
        print(f"   - Method: {self.vpt_selector.selection_method}")
        print(f"   - Selected features: {self.vpt_selector.n_features}")
        print(f"\n4. 3D CNN:")
        print(f"   - Input shape: {self.cnn_model.input_shape}")
        print(f"   - Classes: {self.num_classes}")
        print(f"   - Device: {self.device}")
        print(f"\n5. 3D U-Net:")
        print(f"   - Input shape: {self.unet_model.input_shape}")
        print(f"   - Base filters: {self.unet_model.base_filters}")
        print(f"   - Device: {self.device}")
        print(f"\n6. Ensemble:")
        print(f"   - Strategy: {self.ensemble_model.strategy}")
        print(f"   - Weights: α={self.ensemble_model.alpha}, β={self.ensemble_model.beta}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    print("Full Pipeline Module - Ready for import")
    print("Use: from full_pipeline import BrainTumorSegmentationPipeline")
