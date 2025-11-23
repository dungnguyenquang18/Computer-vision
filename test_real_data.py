# coding: utf-8
"""
Test pipeline with real BraTS data
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import pipeline
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import BrainTumorSegmentationPipeline

# Import nibabel for loading NIfTI files
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[ERROR] nibabel not installed. Install with: pip install nibabel")
    sys.exit(1)


def load_brats_patient(data_dir, patient_id="BraTS2021_00451"):
    """
    Load BraTS patient data (4 modalities)
    
    Args:
        data_dir: Directory containing .nii.gz files
        patient_id: Patient ID
        
    Returns:
        volume: (H, W, D, 4) numpy array with 4 modalities
        ground_truth: (H, W, D) segmentation mask (if available)
    """
    data_dir = Path(data_dir)
    
    # Define modality names
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    print(f"\n{'='*80}")
    print(f"Loading BraTS patient: {patient_id}")
    print(f"{'='*80}")
    
    # Load each modality
    volume_list = []
    for modality in modalities:
        filepath = data_dir / f"{patient_id}_{modality}.nii.gz"
        
        if not filepath.exists():
            print(f"[WARNING] File not found: {filepath}")
            # Create dummy data
            volume_list.append(np.zeros((240, 240, 155)))
        else:
            print(f"Loading {modality}: {filepath.name}")
            nifti = nib.load(str(filepath))
            data = nifti.get_fdata()
            print(f"  Shape: {data.shape}, Range: [{data.min():.2f}, {data.max():.2f}]")
            volume_list.append(data)
    
    # Stack along channel dimension
    volume = np.stack(volume_list, axis=-1).astype(np.float32)
    print(f"\n[OK] Combined volume shape: {volume.shape}")
    
    # Load ground truth if available
    seg_filepath = data_dir / f"{patient_id}_seg.nii.gz"
    ground_truth = None
    if seg_filepath.exists():
        print(f"\nLoading ground truth: {seg_filepath.name}")
        nifti = nib.load(str(seg_filepath))
        ground_truth = nifti.get_fdata().astype(np.uint8)
        print(f"  Shape: {ground_truth.shape}")
        print(f"  Unique labels: {np.unique(ground_truth)}")
    
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


def evaluate_segmentation(pred, gt):
    """Evaluate segmentation against ground truth"""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    
    # BraTS labels: 0=Background, 1=Necrotic, 2=Edema, 4=Enhancing
    labels = {
        0: "Background",
        1: "Necrotic Core",
        2: "Edema",
        4: "Enhancing Tumor"
    }
    
    print(f"\nPrediction shape: {pred.shape}")
    print(f"Ground truth shape: {gt.shape}")
    print(f"\nPredicted labels: {np.unique(pred)}")
    print(f"Ground truth labels: {np.unique(gt)}")
    
    # Calculate Dice scores
    print(f"\n{'Label':<20} {'Dice Score':>12}")
    print("-" * 35)
    
    dice_scores = {}
    for label, name in labels.items():
        if label in np.unique(gt):
            dice = calculate_dice_score(pred, gt, label)
            dice_scores[label] = dice
            print(f"{name:<20} {dice:>12.4f}")
    
    # Calculate mean Dice (excluding background)
    tumor_labels = [1, 2, 4]
    tumor_dices = [dice_scores.get(l, 0) for l in tumor_labels if l in dice_scores]
    if tumor_dices:
        mean_dice = np.mean(tumor_dices)
        print("-" * 35)
        print(f"{'Mean Dice (Tumor)':<20} {mean_dice:>12.4f}")
    
    # Volume statistics
    print(f"\n{'='*80}")
    print("VOLUME STATISTICS")
    print(f"{'='*80}")
    
    total_voxels = pred.size
    print(f"\n{'Label':<20} {'Predicted':>12} {'Ground Truth':>15} {'% Diff':>10}")
    print("-" * 60)
    
    for label, name in labels.items():
        pred_count = np.sum(pred == label)
        gt_count = np.sum(gt == label)
        
        if gt_count > 0:
            diff = ((pred_count - gt_count) / gt_count) * 100
            print(f"{name:<20} {pred_count:>12} {gt_count:>15} {diff:>9.1f}%")
    
    print(f"{'='*80}\n")
    
    return dice_scores


def test_with_real_data():
    """Test pipeline with real BraTS data"""
    
    print("\n" + "#"*80)
    print("#  TESTING PIPELINE WITH REAL BRATS DATA  #".center(80))
    print("#"*80)
    
    # Load data
    data_dir = Path(__file__).parent.parent
    volume, ground_truth = load_brats_patient(data_dir, "BraTS2021_00451")
    
    print(f"Input volume shape: {volume.shape}")
    print(f"Expected: (240, 240, 155, 4)")
    
    if volume.shape != (240, 240, 155, 4):
        print(f"[WARNING] Volume shape mismatch! Got {volume.shape}")
        # Reshape if needed
        if volume.shape[2] != 155:
            print(f"[INFO] Adjusting depth dimension...")
    
    # Initialize pipeline with appropriate settings
    print("\n" + "="*80)
    print("INITIALIZING PIPELINE")
    print("="*80)
    
    pipeline = BrainTumorSegmentationPipeline(
        target_shape=(240, 240, 160, 4),  # Standard BraTS with padding
        glcm_window_size=5,                # Larger window for real data
        glcm_levels=16,                    # More levels for better features
        vpt_n_features=10,                 # Keep more features
        vpt_method='variance',
        patch_size=(128, 128, 128),        # Standard patch size
        num_classes=4,
        ensemble_strategy='weighted',
        ensemble_alpha=0.4,
        ensemble_beta=0.6,
        device='cpu'  # Change to 'cuda' if you have GPU
    )
    
    pipeline.get_pipeline_summary()
    
    # Run pipeline
    print("\n" + "#"*80)
    print("#  RUNNING SEGMENTATION PIPELINE  #".center(80))
    print("#"*80)
    print("\n[WARNING] This will take some time with real data...")
    print("[INFO] Using fast mode with large strides to speed up testing\n")
    
    try:
        # Run WITHOUT fast mode to preserve dimensions
        # Fast mode reduces spatial dimensions which causes mismatch
        print("[INFO] Running in standard mode (no fast GLCM)")
        print("[INFO] This will take longer but preserves spatial dimensions\n")
        
        prediction = pipeline.predict(
            volume,
            fast_mode=False,         # Use standard GLCM to preserve dimensions
            glcm_stride=1,           # Not used when fast_mode=False
            pred_stride=(128, 128, 128)  # No overlap for speed
        )
        
        print(f"\n[OK] Segmentation completed!")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Unique labels in prediction: {np.unique(prediction)}")
        
        # Save prediction
        output_path = Path(__file__).parent / "prediction_BraTS2021_00451.npy"
        np.save(output_path, prediction)
        print(f"\n[OK] Prediction saved to: {output_path}")
        
        # Evaluate if ground truth available
        if ground_truth is not None:
            dice_scores = evaluate_segmentation(prediction, ground_truth)
            
            # Save evaluation results
            eval_path = Path(__file__).parent / "evaluation_results.txt"
            with open(eval_path, 'w') as f:
                f.write("BraTS Segmentation Evaluation Results\n")
                f.write("="*50 + "\n\n")
                for label, dice in dice_scores.items():
                    label_name = {0: "Background", 1: "Necrotic", 2: "Edema", 4: "Enhancing"}
                    f.write(f"{label_name.get(label, f'Label {label}')}: {dice:.4f}\n")
            print(f"[OK] Evaluation saved to: {eval_path}")
        else:
            print("\n[INFO] No ground truth available for evaluation")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU - this will be slower")
    
    # Run test
    success = test_with_real_data()
    
    if success:
        print("\n" + "="*80)
        print("  TEST COMPLETED SUCCESSFULLY!".center(80))
        print("="*80)
    else:
        print("\n" + "="*80)
        print("  TEST FAILED!".center(80))
        print("="*80)
