"""
Post-processing utilities cho segmentation masks
Bao gồm: unpadding, morphological operations, consistency checks
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def unpad_volume(
    volume: torch.Tensor,
    original_depth: int = 155,
    padded_depth: int = 160
) -> torch.Tensor:
    """
    Cắt bỏ padding để về kích thước gốc
    Từ [C, 160, H, W] về [C, 155, H, W]
    
    Args:
        volume: Volume đã được pad [C, D_padded, H, W] hoặc [D_padded, H, W]
        original_depth: Độ sâu gốc (155)
        padded_depth: Độ sâu sau khi pad (160)
        
    Returns:
        volume_unpadded: Volume đã được unpad
    """
    if volume.dim() == 4:
        # [C, D, H, W]
        pad_total = padded_depth - original_depth
        pad_start = pad_total // 2
        pad_end = pad_start + original_depth
        return volume[:, pad_start:pad_end, :, :]
    elif volume.dim() == 3:
        # [D, H, W]
        pad_total = padded_depth - original_depth
        pad_start = pad_total // 2
        pad_end = pad_start + original_depth
        return volume[pad_start:pad_end, :, :]
    else:
        raise ValueError(f"Unsupported volume dimension: {volume.dim()}")


def remove_small_components(
    mask: torch.Tensor,
    min_size: int = 100,
    connectivity: int = 3
) -> torch.Tensor:
    """
    Loại bỏ các connected components nhỏ (nhiễu)
    Sử dụng connected component analysis
    
    Args:
        mask: Segmentation mask [D, H, W]
        min_size: Kích thước tối thiểu (voxels) để giữ lại
        connectivity: Độ kết nối (1, 2, hoặc 3 cho 3D)
        
    Returns:
        mask_cleaned: Mask đã được làm sạch
    """
    # Chuyển sang numpy để xử lý
    device = mask.device
    mask_np = mask.cpu().numpy()
    mask_cleaned = np.zeros_like(mask_np)
    
    # Xử lý từng class (không bao gồm background = 0)
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        # Tạo binary mask cho class này
        binary_mask = (mask_np == label).astype(np.uint8)
        
        # Connected component analysis
        labeled_array, num_features = ndimage.label(
            binary_mask,
            structure=ndimage.generate_binary_structure(3, connectivity)
        )
        
        # Đếm kích thước mỗi component
        for i in range(1, num_features + 1):
            component = (labeled_array == i)
            component_size = np.sum(component)
            
            # Giữ lại nếu đủ lớn
            if component_size >= min_size:
                mask_cleaned[component] = label
    
    return torch.from_numpy(mask_cleaned).to(device)


def morphological_closing(
    mask: torch.Tensor,
    kernel_size: int = 3,
    iterations: int = 1
) -> torch.Tensor:
    """
    Morphological closing để làm mịn ranh giới và lấp các lỗ nhỏ
    Closing = Dilation followed by Erosion
    
    Args:
        mask: Segmentation mask [D, H, W]
        kernel_size: Kích thước kernel (3, 5, ...)
        iterations: Số lần lặp lại operation
        
    Returns:
        mask_closed: Mask sau khi closing
    """
    device = mask.device
    mask_np = mask.cpu().numpy()
    mask_closed = np.zeros_like(mask_np)
    
    # Tạo structuring element (sphere)
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, kernel_size // 2)
    
    # Xử lý từng class
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        binary_mask = (mask_np == label).astype(np.uint8)
        
        # Closing operation
        closed = ndimage.binary_closing(
            binary_mask,
            structure=struct,
            iterations=iterations
        )
        
        mask_closed[closed > 0] = label
    
    return torch.from_numpy(mask_closed).to(device)


def fill_holes(
    mask: torch.Tensor,
    max_hole_size: int = 1000
) -> torch.Tensor:
    """
    Lấp các lỗ (holes) trong segmentation mask
    
    Args:
        mask: Segmentation mask [D, H, W]
        max_hole_size: Kích thước tối đa của hole để lấp (voxels)
        
    Returns:
        mask_filled: Mask đã được lấp holes
    """
    device = mask.device
    mask_np = mask.cpu().numpy()
    mask_filled = mask_np.copy()
    
    # Xử lý từng class
    unique_labels = np.unique(mask_np)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        binary_mask = (mask_np == label).astype(bool)
        
        # Fill holes
        filled = ndimage.binary_fill_holes(binary_mask)
        
        # Chỉ lấp các holes không quá lớn
        holes = filled & ~binary_mask
        labeled_holes, num_holes = ndimage.label(holes)
        
        for i in range(1, num_holes + 1):
            hole = (labeled_holes == i)
            hole_size = np.sum(hole)
            
            if hole_size <= max_hole_size:
                mask_filled[hole] = label
    
    return torch.from_numpy(mask_filled).to(device)


def enforce_consistency(
    mask: torch.Tensor,
    rules: Optional[dict] = None
) -> torch.Tensor:
    """
    Đảm bảo tính nhất quán logic của segmentation
    Ví dụ: necrotic core (1) phải nằm trong tumor (4)
    
    BraTS labels:
    - 0: Background
    - 1: Necrotic core (NCR)
    - 2: Edema (ED)
    - 4: Enhancing tumor (ET)
    
    Rules:
    - NCR phải có ít nhất một voxel ET hoặc ED lân cận
    - ET thường được bao quanh bởi ED
    
    Args:
        mask: Segmentation mask [D, H, W]
        rules: Custom rules (optional)
        
    Returns:
        mask_consistent: Mask đã được enforce consistency
    """
    device = mask.device
    mask_np = mask.cpu().numpy()
    mask_consistent = mask_np.copy()
    
    # Rule 1: NCR (1) không nên tồn tại độc lập, phải gần ET (4) hoặc ED (2)
    ncr_mask = (mask_np == 1)
    if np.any(ncr_mask):
        # Dilate tumor regions (ED + ET)
        tumor_mask = (mask_np == 2) | (mask_np == 4)
        dilated_tumor = ndimage.binary_dilation(tumor_mask, iterations=2)
        
        # NCR ngoài tumor region → chuyển thành background
        isolated_ncr = ncr_mask & ~dilated_tumor
        mask_consistent[isolated_ncr] = 0
    
    # Rule 2: Các vùng ET (4) rất nhỏ và cô lập → có thể là noise
    et_mask = (mask_np == 4)
    if np.any(et_mask):
        labeled_et, num_et = ndimage.label(et_mask)
        for i in range(1, num_et + 1):
            component = (labeled_et == i)
            if np.sum(component) < 50:  # Quá nhỏ
                # Check xem có nằm trong vùng ED không
                dilated_ed = ndimage.binary_dilation(mask_np == 2, iterations=1)
                if not np.any(component & dilated_ed):
                    mask_consistent[component] = 0
    
    return torch.from_numpy(mask_consistent).to(device)


def postprocess_mask(
    mask: torch.Tensor,
    original_depth: int = 155,
    padded_depth: int = 160,
    remove_small: bool = True,
    min_component_size: int = 100,
    smooth_boundary: bool = True,
    kernel_size: int = 3,
    fill_holes_flag: bool = True,
    max_hole_size: int = 1000,
    enforce_consistency_flag: bool = True
) -> torch.Tensor:
    """
    Pipeline post-processing đầy đủ cho segmentation mask
    
    Args:
        mask: Segmentation mask [D, H, W] với D=160 (padded)
        original_depth: Độ sâu gốc (155)
        padded_depth: Độ sâu sau khi pad (160)
        remove_small: Có loại bỏ small components không
        min_component_size: Kích thước tối thiểu của component
        smooth_boundary: Có làm mịn ranh giới không
        kernel_size: Kích thước kernel cho morphological ops
        fill_holes_flag: Có lấp holes không
        max_hole_size: Kích thước tối đa của hole
        enforce_consistency_flag: Có enforce consistency không
        
    Returns:
        mask_processed: Mask đã được post-process [155, H, W]
    """
    # 1. Unpadding (160 → 155)
    mask = unpad_volume(mask, original_depth, padded_depth)
    
    # 2. Remove small components
    if remove_small:
        mask = remove_small_components(mask, min_size=min_component_size)
    
    # 3. Fill holes
    if fill_holes_flag:
        mask = fill_holes(mask, max_hole_size=max_hole_size)
    
    # 4. Smooth boundary (morphological closing)
    if smooth_boundary:
        mask = morphological_closing(mask, kernel_size=kernel_size)
    
    # 5. Enforce consistency
    if enforce_consistency_flag:
        mask = enforce_consistency(mask)
    
    return mask


def postprocess_probabilities(
    prob: torch.Tensor,
    original_depth: int = 155,
    padded_depth: int = 160
) -> torch.Tensor:
    """
    Post-processing cho probability maps (trước khi argmax)
    Chủ yếu là unpadding
    
    Args:
        prob: Probability map [C, D, H, W] với D=160
        original_depth: Độ sâu gốc (155)
        padded_depth: Độ sâu sau khi pad (160)
        
    Returns:
        prob_unpadded: [C, 155, H, W]
    """
    return unpad_volume(prob, original_depth, padded_depth)
