"""
Module Trích Xuất Đặc Trưng GLCM với PyTorch (GPU-accelerated)
Trích xuất các đặc trưng kết cấu từ ảnh MRI sử dụng Grey Level Co-occurrence Matrix.
"""

import torch
import torch.nn.functional as F
import numpy as np
import warnings


class GLCMExtractorTorch:
    """
    Class trích xuất đặc trưng GLCM từ volume MRI với PyTorch để tận dụng GPU.
    
    Chức năng:
    - Tính toán GLCM (Grey Level Co-occurrence Matrix) trên GPU
    - Trích xuất 5 đặc trưng Haralick cho mỗi kênh
    - Output: 4 kênh input → 20 kênh features
    - Hỗ trợ batch processing
    
    Attributes:
        window_size (int): Kích thước cửa sổ trượt (sliding window)
        levels (int): Số mức xám cho quantization
        device (str): 'cuda' hoặc 'cpu'
    """
    
    def __init__(self, window_size=5, levels=16, device='cuda'):
        """
        Khởi tạo GLCMExtractorTorch.
        
        Args:
            window_size (int): Kích thước cửa sổ trượt (3, 5, 7, ...)
            levels (int): Số mức xám cho quantization (8, 16, 32)
            device (str): 'cuda' hoặc 'cpu'
        """
        self.window_size = window_size
        self.levels = levels
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 5 đặc trưng Haralick
        self.feature_names = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        print(f"[GLCMExtractor-Torch] Initialized:")
        print(f"  - Window size: {window_size}×{window_size}×{window_size}")
        print(f"  - Gray levels: {levels}")
        print(f"  - Features: {len(self.feature_names)} per channel")
        print(f"  - Device: {self.device}")
    
    def _quantize_image_torch(self, image_tensor):
        """
        Quantize ảnh về số mức xám cố định (GPU).
        
        Args:
            image_tensor (torch.Tensor): Ảnh đầu vào
            
        Returns:
            torch.Tensor: Ảnh đã quantize (dtype uint8 hoặc long)
        """
        # Normalize về [0, 1]
        img_min = image_tensor.min()
        img_max = image_tensor.max()
        
        if img_max - img_min < 1e-8:
            return torch.zeros_like(image_tensor, dtype=torch.long)
        
        normalized = (image_tensor - img_min) / (img_max - img_min)
        
        # Quantize về levels mức
        quantized = (normalized * (self.levels - 1)).long()
        quantized = torch.clamp(quantized, 0, self.levels - 1)
        
        return quantized
    
    def compute_glcm_torch(self, patch, direction='horizontal'):
        """
        Tính GLCM cho một patch sử dụng PyTorch (GPU).
        
        Args:
            patch (torch.Tensor): Patch đã quantize với shape (H, W)
            direction (str): 'horizontal', 'vertical', 'diagonal', 'antidiagonal'
            
        Returns:
            torch.Tensor: GLCM matrix (levels × levels)
        """
        # Shift theo direction
        if direction == 'horizontal':
            p1 = patch[:, :-1]
            p2 = patch[:, 1:]
        elif direction == 'vertical':
            p1 = patch[:-1, :]
            p2 = patch[1:, :]
        elif direction == 'diagonal':
            p1 = patch[:-1, :-1]
            p2 = patch[1:, 1:]
        elif direction == 'antidiagonal':
            p1 = patch[:-1, 1:]
            p2 = patch[1:, :-1]
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Flatten
        p1_flat = p1.flatten()
        p2_flat = p2.flatten()
        
        # Build GLCM using bincount (fast on GPU)
        glcm = torch.zeros((self.levels, self.levels), dtype=torch.float32, device=self.device)
        
        # Compute co-occurrence
        indices = p1_flat * self.levels + p2_flat
        glcm_flat = torch.bincount(indices, minlength=self.levels * self.levels)
        glcm = glcm_flat[:self.levels * self.levels].reshape(self.levels, self.levels).float()
        
        # Make symmetric
        glcm = glcm + glcm.T
        
        # Normalize
        glcm_sum = glcm.sum()
        if glcm_sum > 0:
            glcm = glcm / glcm_sum
        
        return glcm
    
    def extract_haralick_features_torch(self, glcm):
        """
        Trích xuất 5 đặc trưng Haralick từ GLCM matrix (GPU).
        
        Args:
            glcm (torch.Tensor): GLCM matrix (levels × levels)
            
        Returns:
            dict: Dictionary chứa 5 giá trị đặc trưng
        """
        features = {}
        
        # Create coordinate matrices
        i_coords = torch.arange(self.levels, dtype=torch.float32, device=self.device)
        j_coords = torch.arange(self.levels, dtype=torch.float32, device=self.device)
        i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
        
        # 1. Contrast
        contrast = ((i_grid - j_grid) ** 2 * glcm).sum()
        features['contrast'] = contrast.item()
        
        # 2. Dissimilarity
        dissimilarity = (torch.abs(i_grid - j_grid) * glcm).sum()
        features['dissimilarity'] = dissimilarity.item()
        
        # 3. Homogeneity (Inverse Difference Moment)
        homogeneity = (glcm / (1 + (i_grid - j_grid) ** 2)).sum()
        features['homogeneity'] = homogeneity.item()
        
        # 4. Energy (Angular Second Moment)
        energy = (glcm ** 2).sum()
        features['energy'] = energy.item()
        
        # 5. Correlation
        mu_i = (i_grid * glcm).sum()
        mu_j = (j_grid * glcm).sum()
        sigma_i = torch.sqrt(((i_grid - mu_i) ** 2 * glcm).sum())
        sigma_j = torch.sqrt(((j_grid - mu_j) ** 2 * glcm).sum())
        
        if sigma_i > 1e-8 and sigma_j > 1e-8:
            correlation = (((i_grid - mu_i) * (j_grid - mu_j) * glcm).sum() / (sigma_i * sigma_j))
            features['correlation'] = correlation.item()
        else:
            features['correlation'] = 0.0
        
        return features
    
    def extract_features_from_slice_torch(self, slice_2d):
        """
        Trích xuất đặc trưng GLCM từ một slice 2D (GPU).
        
        Args:
            slice_2d (torch.Tensor): Slice 2D với shape (H, W)
            
        Returns:
            torch.Tensor: Feature maps với shape (H, W, 5)
        """
        h, w = slice_2d.shape
        half_window = self.window_size // 2
        
        # Initialize feature maps
        feature_maps = torch.zeros((h, w, len(self.feature_names)), 
                                   dtype=torch.float32, device=self.device)
        
        # Quantize slice
        quantized = self._quantize_image_torch(slice_2d)
        
        # Sliding window (vectorized approach for speed)
        directions = ['horizontal', 'vertical', 'diagonal', 'antidiagonal']
        
        for i in range(half_window, h - half_window):
            for j in range(half_window, w - half_window):
                # Extract patch
                patch = quantized[
                    i - half_window:i + half_window + 1,
                    j - half_window:j + half_window + 1
                ]
                
                # Average features across all directions
                avg_features = {name: 0.0 for name in self.feature_names}
                
                for direction in directions:
                    glcm = self.compute_glcm_torch(patch, direction)
                    features = self.extract_haralick_features_torch(glcm)
                    
                    for name in self.feature_names:
                        avg_features[name] += features[name]
                
                # Average
                for name in self.feature_names:
                    avg_features[name] /= len(directions)
                
                # Store features
                for feat_idx, feat_name in enumerate(self.feature_names):
                    feature_maps[i, j, feat_idx] = avg_features[feat_name]
        
        return feature_maps
    
    def extract_features_channel_torch(self, channel_volume):
        """
        Trích xuất đặc trưng GLCM cho một kênh 3D (GPU).
        
        Args:
            channel_volume (torch.Tensor): Volume 3D single channel (H, W, D)
            
        Returns:
            torch.Tensor: Feature volume với shape (H, W, D, 5)
        """
        h, w, d = channel_volume.shape
        feature_volume = torch.zeros((h, w, d, len(self.feature_names)), 
                                    dtype=torch.float32, device=self.device)
        
        # Process each slice
        for z in range(d):
            slice_2d = channel_volume[:, :, z]
            feature_maps = self.extract_features_from_slice_torch(slice_2d)
            feature_volume[:, :, z, :] = feature_maps
            
            # Progress indicator
            if (z + 1) % 20 == 0:
                print(f"  {z+1}/{d}", end=' ', flush=True)
        
        print()
        return feature_volume
    
    def extract_features(self, volume):
        """
        Trích xuất đặc trưng GLCM cho toàn bộ volume 4D.
        
        Input: (H, W, D, C) - C channels (NumPy)
        Output: (H, W, D, C*5) - C*5 feature channels (NumPy)
        
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
        print(f"[GLCMExtractor-Torch] Starting feature extraction (GPU)...")
        print(f"{'='*60}")
        print(f"Input shape: {volume.shape}")
        print(f"Output will be: ({h}, {w}, {d}, {num_channels * num_features})")
        
        # Convert to torch tensor
        volume_tensor = torch.from_numpy(volume).float().to(self.device)
        
        # Initialize output tensor
        feature_tensor = torch.zeros(
            (h, w, d, num_channels * num_features),
            dtype=torch.float32,
            device=self.device
        )
        
        # Extract features for each channel
        for c in range(num_channels):
            print(f"\n[GLCMExtractor-Torch] Processing channel {c+1}/{num_channels}...")
            
            channel_volume = volume_tensor[:, :, :, c]
            channel_features = self.extract_features_channel_torch(channel_volume)
            
            # Store in output tensor
            start_idx = c * num_features
            end_idx = start_idx + num_features
            feature_tensor[:, :, :, start_idx:end_idx] = channel_features
            
            # Statistics
            for feat_idx, feat_name in enumerate(self.feature_names):
                feat_data = channel_features[:, :, :, feat_idx]
                print(f"    {feat_name}: mean={feat_data.mean():.4f}, std={feat_data.std():.4f}")
        
        print(f"\n[GLCMExtractor-Torch] Feature extraction completed!")
        print(f"Final shape: {feature_tensor.shape}")
        print(f"{'='*60}\n")
        
        # Convert back to numpy
        return feature_tensor.cpu().numpy()
    
    def extract_features_fast(self, volume, stride=2):
        """
        Version nhanh hơn với stride (GPU).
        
        Args:
            volume (numpy.ndarray): Volume với shape (H, W, D, C)
            stride (int): Stride cho sliding window
            
        Returns:
            numpy.ndarray: Feature tensor (có thể nhỏ hơn nếu stride > 1)
        """
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume (H, W, D, C), got shape {volume.shape}")
        
        h, w, d, num_channels = volume.shape
        num_features = len(self.feature_names)
        
        print(f"\n[GLCMExtractor-Torch] Fast mode (stride={stride}, GPU)...")
        
        # Calculate output size with stride
        out_h = (h + stride - 1) // stride
        out_w = (w + stride - 1) // stride
        
        # Convert to torch
        volume_tensor = torch.from_numpy(volume).float().to(self.device)
        
        feature_tensor = torch.zeros(
            (out_h, out_w, d, num_channels * num_features),
            dtype=torch.float32,
            device=self.device
        )
        
        half_window = self.window_size // 2
        directions = ['horizontal', 'vertical', 'diagonal', 'antidiagonal']
        
        for c in range(num_channels):
            print(f"Channel {c+1}/{num_channels}...", end=' ', flush=True)
            channel_volume = volume_tensor[:, :, :, c]
            
            for z in range(d):
                quantized = self._quantize_image_torch(channel_volume[:, :, z])
                
                out_i = 0
                for i in range(half_window, h - half_window, stride):
                    out_j = 0
                    for j in range(half_window, w - half_window, stride):
                        patch = quantized[
                            i - half_window:i + half_window + 1,
                            j - half_window:j + half_window + 1
                        ]
                        
                        # Average across directions
                        avg_features = {name: 0.0 for name in self.feature_names}
                        for direction in directions:
                            glcm = self.compute_glcm_torch(patch, direction)
                            features = self.extract_haralick_features_torch(glcm)
                            for name in self.feature_names:
                                avg_features[name] += features[name]
                        
                        for name in self.feature_names:
                            avg_features[name] /= len(directions)
                        
                        start_idx = c * num_features
                        for feat_idx, feat_name in enumerate(self.feature_names):
                            feature_tensor[out_i, out_j, z, start_idx + feat_idx] = avg_features[feat_name]
                        
                        out_j += 1
                    out_i += 1
            
            print("Done!")
        
        print(f"Output shape: {feature_tensor.shape}\n")
        return feature_tensor.cpu().numpy()


if __name__ == "__main__":
    # Quick test
    print("Running GPU test...")
    
    extractor = GLCMExtractorTorch(window_size=3, levels=16, device='cuda')
    
    # Create small synthetic volume
    small_volume = np.random.randn(50, 50, 10, 2).astype(np.float32)
    print(f"\nSynthetic volume shape: {small_volume.shape}")
    
    # Test fast extraction
    features = extractor.extract_features_fast(small_volume, stride=5)
    print(f"Features shape: {features.shape}")
    print(f"Value range: [{features.min():.4f}, {features.max():.4f}]")
