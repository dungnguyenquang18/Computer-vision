"""
Module Trích Xuất Đặc Trưng GLCM (GLCM Feature Extraction)
Trích xuất các đặc trưng kết cấu từ ảnh MRI sử dụng Grey Level Co-occurrence Matrix.
"""

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import graycomatrix, graycoprops
import warnings


class GLCMExtractor:
    """
    Class trích xuất đặc trưng GLCM từ volume MRI đã được tiền xử lý.
    
    Chức năng:
    - Tính toán GLCM (Grey Level Co-occurrence Matrix)
    - Trích xuất 5 đặc trưng Haralick cho mỗi kênh
    - Output: 4 kênh input → 20 kênh features
    
    Attributes:
        window_size (int): Kích thước cửa sổ trượt (sliding window)
        distances (list): Khoảng cách cho GLCM computation
        angles (list): Các góc tính GLCM (radians)
        levels (int): Số mức xám cho quantization
    """
    
    def __init__(self, window_size=5, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32):
        """
        Khởi tạo GLCMExtractor.
        
        Args:
            window_size (int): Kích thước cửa sổ trượt (3, 5, 7, ...)
            distances (list): Khoảng cách tính GLCM, mặc định [1]
            angles (list): Góc tính GLCM (rad), mặc định [0°, 45°, 90°, 135°]
            levels (int): Số mức xám cho quantization (8, 16, 32, 64)
        """
        self.window_size = window_size
        self.distances = distances
        self.angles = angles
        self.levels = levels
        
        # 5 đặc trưng Haralick sẽ trích xuất
        self.feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        print(f"[GLCMExtractor] Initialized:")
        print(f"  - Window size: {window_size}×{window_size}×{window_size}")
        print(f"  - Distances: {distances}")
        print(f"  - Angles: {len(angles)} directions")
        print(f"  - Gray levels: {levels}")
        print(f"  - Features: {len(self.feature_names)} per channel")
    
    def _quantize_image(self, image):
        """
        Quantize ảnh về số mức xám cố định để tính GLCM.
        
        Args:
            image (numpy.ndarray): Ảnh đầu vào (đã normalized)
            
        Returns:
            numpy.ndarray: Ảnh đã quantize (dtype uint8)
        """
        # Normalize về [0, 1]
        img_min = image.min()
        img_max = image.max()
        
        if img_max - img_min < 1e-8:
            # Constant image
            return np.zeros_like(image, dtype=np.uint8)
        
        normalized = (image - img_min) / (img_max - img_min)
        
        # Quantize về levels mức
        quantized = (normalized * (self.levels - 1)).astype(np.uint8)
        
        return quantized
    
    def compute_glcm_2d(self, patch_2d):
        """
        Tính GLCM cho một patch 2D.
        
        Args:
            patch_2d (numpy.ndarray): Patch 2D đã quantize
            
        Returns:
            numpy.ndarray: GLCM matrix
        """
        if patch_2d.size == 0:
            return None
        
        # Tính GLCM với skimage
        glcm = graycomatrix(
            patch_2d,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )
        
        return glcm
    
    def extract_haralick_features_2d(self, glcm):
        """
        Trích xuất 5 đặc trưng Haralick từ GLCM matrix.
        
        5 features:
        1. Contrast (Differentiation - Độ tương phản)
        2. Dissimilarity (Divergence - Phân kỳ)
        3. Homogeneity (Đồng nhất)
        4. Energy (Năng lượng)
        5. Correlation (Relationship - Quan hệ)
        
        Args:
            glcm (numpy.ndarray): GLCM matrix
            
        Returns:
            dict: Dictionary chứa 5 giá trị đặc trưng
        """
        if glcm is None:
            return {name: 0.0 for name in self.feature_names}
        
        features = {}
        
        try:
            # Tính từng đặc trưng và average across directions
            for prop in self.feature_names:
                values = graycoprops(glcm, prop)
                # Average over distances and angles
                features[prop] = np.mean(values)
        except Exception as e:
            warnings.warn(f"Error computing GLCM features: {e}")
            features = {name: 0.0 for name in self.feature_names}
        
        return features
    
    def extract_features_from_slice(self, slice_2d):
        """
        Trích xuất đặc trưng GLCM từ một slice 2D sử dụng sliding window.
        
        Args:
            slice_2d (numpy.ndarray): Slice 2D với shape (H, W)
            
        Returns:
            numpy.ndarray: Feature maps với shape (H, W, 5)
        """
        h, w = slice_2d.shape
        half_window = self.window_size // 2
        
        # Initialize feature maps
        feature_maps = np.zeros((h, w, len(self.feature_names)), dtype=np.float32)
        
        # Quantize slice
        quantized = self._quantize_image(slice_2d)
        
        # Sliding window approach
        for i in range(half_window, h - half_window):
            for j in range(half_window, w - half_window):
                # Extract patch
                patch = quantized[
                    i - half_window:i + half_window + 1,
                    j - half_window:j + half_window + 1
                ]
                
                # Compute GLCM and extract features
                glcm = self.compute_glcm_2d(patch)
                features = self.extract_haralick_features_2d(glcm)
                
                # Store features
                for feat_idx, feat_name in enumerate(self.feature_names):
                    feature_maps[i, j, feat_idx] = features[feat_name]
        
        return feature_maps
    
    def extract_features_channel(self, channel_volume):
        """
        Trích xuất đặc trưng GLCM cho một kênh 3D.
        
        Args:
            channel_volume (numpy.ndarray): Volume 3D single channel (H, W, D)
            
        Returns:
            numpy.ndarray: Feature volume với shape (H, W, D, 5)
        """
        h, w, d = channel_volume.shape
        feature_volume = np.zeros((h, w, d, len(self.feature_names)), dtype=np.float32)
        
        print(f"  [GLCMExtractor] Processing {d} slices...", end=' ')
        
        # Process each slice
        for z in range(d):
            slice_2d = channel_volume[:, :, z]
            feature_maps = self.extract_features_from_slice(slice_2d)
            feature_volume[:, :, z, :] = feature_maps
            
            # Progress indicator
            if (z + 1) % 40 == 0:
                print(f"{z+1}/{d}", end=' ')
        
        print("Done!")
        return feature_volume
    
    def extract_features(self, volume):
        """
        Trích xuất đặc trưng GLCM cho toàn bộ volume 4D.
        
        Input: (H, W, D, C) - C channels
        Output: (H, W, D, C*5) - C*5 feature channels
        
        Args:
            volume (numpy.ndarray): Volume với shape (H, W, D, C)
            
        Returns:
            numpy.ndarray: Feature tensor với shape (H, W, D, C*5)
        """
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume (H, W, D, C), got shape {volume.shape}")
        
        h, w, d, num_channels = volume.shape
        num_features = len(self.feature_names)
        
        print(f"\n{'='*60}")
        print(f"[GLCMExtractor] Starting feature extraction...")
        print(f"{'='*60}")
        print(f"Input shape: {volume.shape}")
        print(f"Output will be: ({h}, {w}, {d}, {num_channels * num_features})")
        
        # Initialize output tensor
        feature_tensor = np.zeros(
            (h, w, d, num_channels * num_features),
            dtype=np.float32
        )
        
        # Extract features for each channel
        for c in range(num_channels):
            print(f"\n[GLCMExtractor] Processing channel {c+1}/{num_channels}...")
            
            channel_volume = volume[:, :, :, c]
            channel_features = self.extract_features_channel(channel_volume)
            
            # Store in output tensor
            start_idx = c * num_features
            end_idx = start_idx + num_features
            feature_tensor[:, :, :, start_idx:end_idx] = channel_features
            
            # Statistics
            for feat_idx, feat_name in enumerate(self.feature_names):
                feat_data = channel_features[:, :, :, feat_idx]
                print(f"    {feat_name}: mean={feat_data.mean():.4f}, std={feat_data.std():.4f}")
        
        print(f"\n[GLCMExtractor] Feature extraction completed!")
        print(f"Final shape: {feature_tensor.shape}")
        print(f"{'='*60}\n")
        
        return feature_tensor
    
    def extract_features_fast(self, volume, stride=2):
        """
        Version nhanh hơn: Subsample sliding window với stride.
        Phù hợp cho testing và prototyping.
        
        Args:
            volume (numpy.ndarray): Volume với shape (H, W, D, C)
            stride (int): Stride cho sliding window (2 = skip every other position)
            
        Returns:
            numpy.ndarray: Feature tensor (có thể nhỏ hơn nếu stride > 1)
        """
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume (H, W, D, C), got shape {volume.shape}")
        
        h, w, d, num_channels = volume.shape
        num_features = len(self.feature_names)
        
        print(f"\n[GLCMExtractor] Fast mode (stride={stride})...")
        
        # Calculate output size with stride
        out_h = (h + stride - 1) // stride
        out_w = (w + stride - 1) // stride
        
        feature_tensor = np.zeros(
            (out_h, out_w, d, num_channels * num_features),
            dtype=np.float32
        )
        
        half_window = self.window_size // 2
        
        for c in range(num_channels):
            print(f"Channel {c+1}/{num_channels}...", end=' ')
            channel_volume = volume[:, :, :, c]
            
            for z in range(d):
                quantized = self._quantize_image(channel_volume[:, :, z])
                
                out_i = 0
                for i in range(half_window, h - half_window, stride):
                    out_j = 0
                    for j in range(half_window, w - half_window, stride):
                        patch = quantized[
                            i - half_window:i + half_window + 1,
                            j - half_window:j + half_window + 1
                        ]
                        
                        glcm = self.compute_glcm_2d(patch)
                        features = self.extract_haralick_features_2d(glcm)
                        
                        start_idx = c * num_features
                        for feat_idx, feat_name in enumerate(self.feature_names):
                            feature_tensor[out_i, out_j, z, start_idx + feat_idx] = features[feat_name]
                        
                        out_j += 1
                    out_i += 1
            
            print("Done!")
        
        print(f"Output shape: {feature_tensor.shape}\n")
        return feature_tensor


if __name__ == "__main__":
    # Quick smoke test
    print("Running smoke test...")
    
    extractor = GLCMExtractor(window_size=3, levels=16)
    
    # Create small synthetic volume (50×50×10×2 for speed)
    small_volume = np.random.randn(50, 50, 10, 2).astype(np.float32)
    print(f"\nSynthetic volume shape: {small_volume.shape}")
    
    # Test fast extraction
    features = extractor.extract_features_fast(small_volume, stride=5)
    print(f"Features shape: {features.shape}")
    print(f"Value range: [{features.min():.4f}, {features.max():.4f}]")
