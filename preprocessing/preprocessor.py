"""
Module Tiền Xử Lý Ảnh MRI (MRI Preprocessing)
Thực hiện zero-padding và chuẩn hóa Z-score cho ảnh MRI đa phương thức.
"""

import numpy as np
from pathlib import Path

# Optional: nibabel for NIfTI file loading (install with: pip install nibabel)
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[Warning] nibabel not installed. NIfTI file loading will not be available.")


class Preprocessor:
    """
    Class tiền xử lý ảnh MRI từ bộ dữ liệu BraTS.
    
    Chức năng:
    - Đọc file NIfTI (.nii hoặc .nii.gz)
    - Zero-padding từ depth 155 → 160
    - Chuẩn hóa Z-score cho từng kênh
    
    Attributes:
        target_shape (tuple): Kích thước đích sau padding (H, W, D, C)
    """
    
    def __init__(self, target_shape=(240, 240, 160, 4)):
        """
        Khởi tạo Preprocessor với kích thước đích.
        
        Args:
            target_shape (tuple): Kích thước đích (height, width, depth, channels)
                                 Mặc định: (240, 240, 160, 4)
        """
        self.target_shape = target_shape
        print(f"[Preprocessor] Initialized with target shape: {target_shape}")
    
    def load_nifti(self, filepath):
        """
        Đọc file NIfTI và trả về numpy array.
        
        Args:
            filepath (str or Path): Đường dẫn đến file .nii hoặc .nii.gz
            
        Returns:
            numpy.ndarray: Dữ liệu ảnh với shape (H, W, D) hoặc (H, W, D, C)
            
        Raises:
            FileNotFoundError: Nếu file không tồn tại
            ValueError: Nếu file không phải định dạng NIfTI hợp lệ
            ImportError: Nếu nibabel chưa được cài đặt
        """
        if not HAS_NIBABEL:
            raise ImportError("nibabel is required for loading NIfTI files. Install with: pip install nibabel")
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix not in ['.nii', '.gz']:
            raise ValueError(f"Invalid file format. Expected .nii or .nii.gz, got {filepath.suffix}")
        
        try:
            nifti_img = nib.load(str(filepath))
            volume = nifti_img.get_fdata()
            print(f"[Preprocessor] Loaded NIfTI file: {filepath.name}, shape: {volume.shape}")
            return volume
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file: {e}")
    
    def pad_volume(self, volume):
        """
        Thêm zero-padding vào chiều depth từ 155 → 160.
        
        Args:
            volume (numpy.ndarray): Volume với shape (H, W, D, C) hoặc (H, W, D)
            
        Returns:
            numpy.ndarray: Volume đã padding với shape target_shape
        """
        # Đảm bảo volume có 4 chiều
        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=-1)
        
        current_shape = volume.shape
        target_h, target_w, target_d, target_c = self.target_shape
        
        # Tính toán padding cho mỗi chiều
        pad_h = max(0, target_h - current_shape[0])
        pad_w = max(0, target_w - current_shape[1])
        pad_d = max(0, target_d - current_shape[2])
        # Chỉ pad channels nếu target yêu cầu nhiều hơn current
        pad_c = max(0, target_c - current_shape[3]) if current_shape[3] < target_c else 0
        
        # Padding đối xứng (thêm đều 2 phía)
        # Đối với depth: thường padding vào cuối
        padded_volume = np.pad(
            volume,
            pad_width=(
                (0, pad_h),           # height: không padding nếu đã đúng
                (0, pad_w),           # width: không padding nếu đã đúng
                (0, pad_d),           # depth: padding từ 155 → 160 (thêm 5 vào cuối)
                (0, pad_c)            # channels: chỉ pad nếu cần thiết
            ),
            mode='constant',
            constant_values=0
        )
        
        print(f"[Preprocessor] Padded volume from {current_shape} to {padded_volume.shape}")
        return padded_volume
    
    def normalize_zscore(self, volume):
        """
        Chuẩn hóa Z-score cho từng kênh độc lập.
        
        Công thức: z = (x - μ) / σ
        - μ: mean của kênh
        - σ: standard deviation của kênh
        
        Args:
            volume (numpy.ndarray): Volume với shape (H, W, D, C)
            
        Returns:
            numpy.ndarray: Volume đã chuẩn hóa với cùng shape
        """
        normalized_volume = np.zeros_like(volume, dtype=np.float32)
        num_channels = volume.shape[-1]
        
        for c in range(num_channels):
            channel_data = volume[:, :, :, c]
            
            # Tính mean và std
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            
            # Tránh chia cho 0
            if std < 1e-8:
                std = 1.0
                print(f"[Warning] Channel {c} has very small std ({std}), using std=1.0")
            
            # Chuẩn hóa Z-score
            normalized_volume[:, :, :, c] = (channel_data - mean) / std
            
            print(f"[Preprocessor] Channel {c}: mean={mean:.4f}, std={std:.4f}")
        
        return normalized_volume
    
    def preprocess(self, volume):
        """
        Pipeline đầy đủ: Padding + Normalization.
        
        Args:
            volume (numpy.ndarray): Volume gốc (H, W, D, C) hoặc (H, W, D)
            
        Returns:
            numpy.ndarray: Volume đã xử lý với shape target_shape
        """
        print(f"\n{'='*60}")
        print(f"[Preprocessor] Starting preprocessing pipeline...")
        print(f"{'='*60}")
        
        # Bước 1: Padding
        padded_volume = self.pad_volume(volume)
        
        # Bước 2: Normalization
        normalized_volume = self.normalize_zscore(padded_volume)
        
        print(f"[Preprocessor] Preprocessing completed!")
        print(f"[Preprocessor] Final shape: {normalized_volume.shape}")
        print(f"{'='*60}\n")
        
        return normalized_volume
    
    def preprocess_from_file(self, filepath):
        """
        Đọc file NIfTI và thực hiện preprocessing đầy đủ.
        
        Args:
            filepath (str or Path): Đường dẫn đến file NIfTI
            
        Returns:
            numpy.ndarray: Volume đã xử lý
        """
        volume = self.load_nifti(filepath)
        return self.preprocess(volume)


if __name__ == "__main__":
    # Quick smoke test với synthetic data
    print("Running smoke test...")
    
    preprocessor = Preprocessor()
    
    # Tạo synthetic volume (240, 240, 155, 4)
    synthetic_volume = np.random.randn(240, 240, 155, 4).astype(np.float32)
    print(f"Synthetic volume shape: {synthetic_volume.shape}")
    
    # Test preprocessing
    processed = preprocessor.preprocess(synthetic_volume)
    print(f"Processed volume shape: {processed.shape}")
    print(f"Value range: [{processed.min():.4f}, {processed.max():.4f}]")
    
    # Verify normalization (should have mean≈0, std≈1 per channel)
    for c in range(4):
        channel_mean = processed[:, :, :, c].mean()
        channel_std = processed[:, :, :, c].std()
        print(f"Channel {c} after normalization: mean={channel_mean:.6f}, std={channel_std:.6f}")
