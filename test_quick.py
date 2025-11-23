# coding: utf-8
"""
Quick test with real BraTS data - using smaller crop for speed
"""

import sys
import numpy as np
import torch
from pathlib import Path
from scipy import ndimage

# Import pipeline
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import BrainTumorSegmentationPipeline

# Import nibabel
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[ERROR] nibabel not installed. Install with: pip install nibabel")
    sys.exit(1)


def load_brats_crop(data_dir, patient_id="BraTS2021_00451", crop_size=64):
    """
    Load a small crop from BraTS data for quick testing
    
    Args:
        data_dir: Directory containing .nii.gz files
        patient_id: Patient ID
        crop_size: Size of crop (default: 64x64x64)
        
    Returns:
        volume: (crop_size, crop_size, crop_size, 4) numpy array
        ground_truth: (crop_size, crop_size, crop_size) segmentation mask
    """
    data_dir = Path(data_dir)
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    print(f"\n{'='*80}")
    print(f"Loading BraTS patient (CROP): {patient_id}")
    print(f"Crop size: {crop_size}x{crop_size}x{crop_size}")
    print(f"{'='*80}")
    
    # Load first modality to determine crop location
    first_file = data_dir / f"{patient_id}_{modalities[0]}.nii.gz"
    nifti = nib.load(str(first_file))
    full_volume = nifti.get_fdata()
    H, W, D = full_volume.shape
    
    print(f"Full volume shape: {full_volume.shape}")
    
    # Find center of mass (where tumor likely is)
    # Use middle of brain as approximation
    center_h = H // 2
    center_w = W // 2
    center_d = D // 2
    
    # Define crop region (centered on tumor area)
    start_h = max(0, center_h - crop_size // 2)
    start_w = max(0, center_w - crop_size // 2)
    start_d = max(0, center_d - crop_size // 2)
    
    end_h = min(H, start_h + crop_size)
    end_w = min(W, start_w + crop_size)
    end_d = min(D, start_d + crop_size)
    
    print(f"Crop region: [{start_h}:{end_h}, {start_w}:{end_w}, {start_d}:{end_d}]")
    
    # Load and crop each modality
    volume_list = []
    for modality in modalities:
        filepath = data_dir / f"{patient_id}_{modality}.nii.gz"
        print(f"Loading {modality}: {filepath.name}")
        
        nifti = nib.load(str(filepath))
        data = nifti.get_fdata()
        
        # Crop
        crop = data[start_h:end_h, start_w:end_w, start_d:end_d]
        volume_list.append(crop)
        print(f"  Crop shape: {crop.shape}, Range: [{crop.min():.2f}, {crop.max():.2f}]")
    
    # Stack
    volume = np.stack(volume_list, axis=-1).astype(np.float32)
    print(f"\n[OK] Combined crop shape: {volume.shape}")
    
    # Load ground truth crop
    seg_filepath = data_dir / f"{patient_id}_seg.nii.gz"
    ground_truth = None
    
    if seg_filepath.exists():
        print(f"\nLoading ground truth crop: {seg_filepath.name}")
        nifti = nib.load(str(seg_filepath))
        gt_full = nifti.get_fdata().astype(np.uint8)
        
        ground_truth = gt_full[start_h:end_h, start_w:end_w, start_d:end_d]
        print(f"  Crop shape: {ground_truth.shape}")
        print(f"  Unique labels: {np.unique(ground_truth)}")
        
        # Check if crop contains tumor
        tumor_voxels = np.sum(ground_truth > 0)
        print(f"  Tumor voxels in crop: {tumor_voxels} ({tumor_voxels/ground_truth.size*100:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return volume, ground_truth


def calculate_dice_score(pred, gt, label):
    """Calculate Dice score for a specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = 2.0 * intersection / union
    return dice


def test_quick():
    """Quick test with cropped data"""
    
    print("\n" + "#"*80)
    print("#  QUICK TEST WITH REAL BRATS DATA (CROPPED)  #".center(80))
    print("#"*80)
    
    # Load cropped data (64x64x64)
    data_dir = Path(__file__).parent.parent
    crop_size = 64
    volume, ground_truth = load_brats_crop(data_dir, "BraTS2021_00451", crop_size=crop_size)
    
    # Initialize pipeline for crop size
    print("\n" + "="*80)
    print("INITIALIZING PIPELINE FOR CROPPED DATA")
    print("="*80)
    
    pipeline = BrainTumorSegmentationPipeline(
        target_shape=(crop_size, crop_size, crop_size, 4),
        glcm_window_size=3,
        glcm_levels=16,
        vpt_n_features=8,
        vpt_method='variance',
        patch_size=(64, 64, 64),  # Single patch = whole crop
        num_classes=4,
        ensemble_strategy='weighted',
        device='cpu'
    )
    
    pipeline.get_pipeline_summary()
    
    # Run pipeline (no fast mode, small data so it's fast anyway)
    print("\n" + "#"*80)
    print("#  RUNNING SEGMENTATION  #".center(80))
    print("#"*80)
    
    try:
        prediction = pipeline.predict(
            volume,
            fast_mode=False,  # Standard mode for accurate dimensions
            pred_stride=(64, 64, 64)  # Single patch
        )
        
        print(f"\n[OK] Segmentation completed!")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Unique labels: {np.unique(prediction)}")
        
        # Evaluate
        if ground_truth is not None:
            print(f"\n{'='*80}")
            print("EVALUATION")
            print(f"{'='*80}")
            
            labels = {0: "Background", 1: "Necrotic", 2: "Edema", 4: "Enhancing"}
            
            print(f"\n{'Label':<20} {'Dice Score':>12} {'GT Voxels':>12} {'Pred Voxels':>12}")
            print("-" * 60)
            
            dice_scores = {}
            for label, name in labels.items():
                if label in np.unique(ground_truth):
                    dice = calculate_dice_score(prediction, ground_truth, label)
                    dice_scores[label] = dice
                    
                    gt_count = np.sum(ground_truth == label)
                    pred_count = np.sum(prediction == label)
                    
                    print(f"{name:<20} {dice:>12.4f} {gt_count:>12} {pred_count:>12}")
            
            # Mean dice (tumor only)
            tumor_labels = [1, 2, 4]
            tumor_dices = [dice_scores.get(l, 0) for l in tumor_labels if l in dice_scores]
            if tumor_dices:
                mean_dice = np.mean(tumor_dices)
                print("-" * 60)
                print(f"{'Mean Dice (Tumor)':<20} {mean_dice:>12.4f}")
            
            print(f"\n[NOTE] Models are UNTRAINED - low scores expected!")
            print(f"[NOTE] This demonstrates the pipeline works end-to-end")
            print(f"[NOTE] Train models on BraTS dataset for good results")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Device: {torch.cuda.get_device_name(0)}")
    
    success = test_quick()
    
    if success:
        print("\n" + "="*80)
        print("  QUICK TEST COMPLETED!".center(80))
        print("="*80)
        print("\n[INFO] Pipeline works correctly with real BraTS data")
        print("[INFO] To get good results, train models on full dataset")
    else:
        print("\n" + "="*80)
        print("  TEST FAILED!".center(80))
        print("="*80)
