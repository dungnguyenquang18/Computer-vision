"""
Module 4: Data Loader for 3D CNN
Handles patch extraction and batching for training and inference.
"""

import numpy as np
from typing import Tuple, Optional, List, Iterator
import warnings

warnings.filterwarnings('ignore')


class PatchDataLoader:
    """
    Data loader for extracting and batching 3D patches from brain MRI volumes.
    
    Handles:
    - Patch extraction from full volumes
    - Data augmentation (optional)
    - Batch generation for training
    - Memory-efficient patch iteration
    """
    
    def __init__(self, patch_size: Tuple[int, int, int] = (128, 128, 128),
                 batch_size: int = 2, shuffle: bool = True):
        """
        Initialize the patch data loader.
        
        Args:
            patch_size: Size of patches to extract (H, W, D)
            batch_size: Number of patches per batch
            shuffle: Whether to shuffle patches during training
        """
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def extract_patches(self, volume: np.ndarray, 
                       mask: Optional[np.ndarray] = None,
                       stride: Optional[Tuple[int, int, int]] = None,
                       random_patches: Optional[int] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract patches from a volume.
        
        Args:
            volume: Input volume of shape (H, W, D, C)
            mask: Optional segmentation mask of shape (H, W, D)
            stride: Stride for patch extraction. If None, uses patch_size
            random_patches: If provided, extract this many random patches instead
            
        Returns:
            Tuple of (patches, patch_masks) where:
            - patches: array of shape (n_patches, pH, pW, pD, C)
            - patch_masks: array of shape (n_patches, pH, pW, pD) or None
        """
        H, W, D = volume.shape[:3]
        pH, pW, pD = self.patch_size
        
        if random_patches:
            # Extract random patches
            return self._extract_random_patches(volume, mask, random_patches)
        
        if stride is None:
            stride = self.patch_size
        
        sH, sW, sD = stride
        
        patches = []
        patch_masks = [] if mask is not None else None
        
        # Extract patches with stride
        for h in range(0, H - pH + 1, sH):
            for w in range(0, W - pW + 1, sW):
                for d in range(0, D - pD + 1, sD):
                    patch = volume[h:h+pH, w:w+pW, d:d+pD, :]
                    patches.append(patch)
                    
                    if mask is not None:
                        patch_mask = mask[h:h+pH, w:w+pW, d:d+pD]
                        patch_masks.append(patch_mask)
        
        patches = np.array(patches)
        if patch_masks:
            patch_masks = np.array(patch_masks)
        
        return patches, patch_masks
    
    def _extract_random_patches(self, volume: np.ndarray,
                               mask: Optional[np.ndarray],
                               n_patches: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract random patches from volume.
        
        Args:
            volume: Input volume of shape (H, W, D, C)
            mask: Optional segmentation mask
            n_patches: Number of patches to extract
            
        Returns:
            Tuple of (patches, patch_masks)
        """
        H, W, D = volume.shape[:3]
        pH, pW, pD = self.patch_size
        
        patches = []
        patch_masks = [] if mask is not None else None
        
        for _ in range(n_patches):
            # Random top-left corner
            h = np.random.randint(0, H - pH + 1)
            w = np.random.randint(0, W - pW + 1)
            d = np.random.randint(0, D - pD + 1)
            
            patch = volume[h:h+pH, w:w+pW, d:d+pD, :]
            patches.append(patch)
            
            if mask is not None:
                patch_mask = mask[h:h+pH, w:w+pW, d:d+pD]
                patch_masks.append(patch_mask)
        
        patches = np.array(patches)
        if patch_masks:
            patch_masks = np.array(patch_masks)
        
        return patches, patch_masks
    
    def create_training_data(self, patches: np.ndarray, 
                            masks: np.ndarray,
                            num_classes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with one-hot encoded labels.
        
        Args:
            patches: Patch array of shape (n_patches, pH, pW, pD, C)
            masks: Mask array of shape (n_patches, pH, pW, pD)
            num_classes: Number of segmentation classes
            
        Returns:
            Tuple of (X, y) where:
            - X: patches (unchanged)
            - y: one-hot encoded labels of shape (n_patches, num_classes)
        """
        # Calculate class distribution for each patch
        n_patches = patches.shape[0]
        y = np.zeros((n_patches, num_classes), dtype=np.float32)
        
        for i in range(n_patches):
            mask = masks[i]
            
            # Map label 4 to index 3
            mask_mapped = np.where(mask == 4, 3, mask)
            
            # Get class distribution
            for class_idx in range(num_classes):
                y[i, class_idx] = np.sum(mask_mapped == class_idx) / mask.size
        
        return patches, y
    
    def augment_patch(self, patch: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply data augmentation to a patch.
        
        Augmentation techniques:
        - Random flips (axis-wise)
        - Random rotations (90, 180, 270 degrees)
        - Small intensity variations
        
        Args:
            patch: Input patch of shape (pH, pW, pD, C)
            mask: Optional mask of shape (pH, pW, pD)
            
        Returns:
            Tuple of (augmented_patch, augmented_mask)
        """
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                patch = np.flip(patch, axis=axis)
                if mask is not None:
                    mask = np.flip(mask, axis=axis)
        
        # Random rotation in xy-plane (k * 90 degrees)
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(0, 1))
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1))
        
        # Random intensity variation (only for image data, not mask)
        if np.random.random() > 0.5:
            intensity_factor = np.random.uniform(0.9, 1.1)
            patch = patch * intensity_factor
        
        # Copy to ensure contiguous arrays
        patch = np.copy(patch)
        if mask is not None:
            mask = np.copy(mask)
        
        return patch, mask
    
    def batch_generator(self, patches: np.ndarray,
                       labels: np.ndarray,
                       augment: bool = False) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate batches for training.
        
        Args:
            patches: Patch array of shape (n_patches, pH, pW, pD, C)
            labels: Label array of shape (n_patches, num_classes)
            augment: Whether to apply data augmentation
            
        Yields:
            Tuples of (batch_patches, batch_labels)
        """
        n_patches = len(patches)
        indices = np.arange(n_patches)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_patches, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_patches)
            batch_indices = indices[start_idx:end_idx]
            
            batch_patches = patches[batch_indices]
            batch_labels = labels[batch_indices]
            
            # Apply augmentation if requested
            if augment:
                augmented_patches = []
                for patch in batch_patches:
                    aug_patch, _ = self.augment_patch(patch)
                    augmented_patches.append(aug_patch)
                batch_patches = np.array(augmented_patches)
            
            yield batch_patches, batch_labels
    
    def normalize_patches(self, patches: np.ndarray, 
                         method: str = 'standard') -> np.ndarray:
        """
        Normalize patch intensities.
        
        Args:
            patches: Patch array of shape (n_patches, pH, pW, pD, C)
            method: Normalization method ('standard', 'minmax', 'z-score')
            
        Returns:
            Normalized patches
        """
        if method == 'standard':
            # Standardize to zero mean and unit variance per patch
            mean = np.mean(patches, axis=(1, 2, 3), keepdims=True)
            std = np.std(patches, axis=(1, 2, 3), keepdims=True)
            patches = (patches - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Scale to [0, 1] range
            min_val = np.min(patches, axis=(1, 2, 3), keepdims=True)
            max_val = np.max(patches, axis=(1, 2, 3), keepdims=True)
            patches = (patches - min_val) / (max_val - min_val + 1e-8)
        
        elif method == 'z-score':
            # Global z-score normalization
            mean = np.mean(patches)
            std = np.std(patches)
            patches = (patches - mean) / (std + 1e-8)
        
        return patches
    
    def balance_classes(self, patches: np.ndarray, 
                       labels: np.ndarray,
                       target_distribution: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance class distribution in the dataset.
        
        Args:
            patches: Patch array
            labels: One-hot encoded labels
            target_distribution: Target class distribution (default: uniform)
            
        Returns:
            Balanced (patches, labels)
        """
        n_classes = labels.shape[1]
        
        if target_distribution is None:
            target_distribution = [1.0 / n_classes] * n_classes
        
        # Get dominant class for each patch
        dominant_classes = np.argmax(labels, axis=1)
        
        # Find indices for each class
        class_indices = [np.where(dominant_classes == i)[0] for i in range(n_classes)]
        
        # Determine target size
        max_samples = max([len(indices) for indices in class_indices])
        
        balanced_patches = []
        balanced_labels = []
        
        for i in range(n_classes):
            indices = class_indices[i]
            n_samples = len(indices)
            
            if n_samples == 0:
                continue
            
            # Oversample or undersample
            target_size = int(max_samples * target_distribution[i])
            
            if n_samples < target_size:
                # Oversample with replacement
                selected_indices = np.random.choice(indices, size=target_size, replace=True)
            else:
                # Undersample
                selected_indices = np.random.choice(indices, size=target_size, replace=False)
            
            balanced_patches.append(patches[selected_indices])
            balanced_labels.append(labels[selected_indices])
        
        balanced_patches = np.vstack(balanced_patches)
        balanced_labels = np.vstack(balanced_labels)
        
        return balanced_patches, balanced_labels
