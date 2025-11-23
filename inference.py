# coding: utf-8
"""
Inference script for Brain Tumor Segmentation
Load trained models and predict on new MRI data
"""

import sys
import numpy as np
import torch
from pathlib import Path
import time

# Add module paths
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import BrainTumorSegmentationPipeline

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[ERROR] nibabel not installed. Install: pip install nibabel")
    sys.exit(1)


def load_mri_volume(data_dir, patient_id):
    """
    Load MRI volume from BraTS dataset
    
    Args:
        data_dir: Directory containing .nii.gz files
        patient_id: Patient ID
        
    Returns:
        volume: (H, W, D, 4) numpy array
    """
    data_dir = Path(data_dir)
    modalities = ['t1', 't1ce', 't2', 'flair']
    
    print(f"\nLoading patient: {patient_id}")
    
    volume_list = []
    for mod in modalities:
        filepath = data_dir / f"{patient_id}_{mod}.nii.gz"
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"  Loading {mod}: {filepath.name}")
        nifti = nib.load(str(filepath))
        data = nifti.get_fdata()
        volume_list.append(data)
    
    volume = np.stack(volume_list, axis=-1).astype(np.float32)
    print(f"  Combined shape: {volume.shape}")
    
    return volume


def save_prediction(prediction, output_path, reference_nifti_path=None):
    """
    Save prediction as NIfTI file
    
    Args:
        prediction: (H, W, D) numpy array
        output_path: Path to save prediction
        reference_nifti_path: Optional path to reference NIfTI for header info
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create NIfTI image
    if reference_nifti_path and Path(reference_nifti_path).exists():
        ref_nifti = nib.load(str(reference_nifti_path))
        pred_nifti = nib.Nifti1Image(prediction, ref_nifti.affine, ref_nifti.header)
    else:
        pred_nifti = nib.Nifti1Image(prediction, np.eye(4))
    
    # Save
    nib.save(pred_nifti, str(output_path))
    print(f"\n[OK] Prediction saved to: {output_path}")


def inference(patient_id, 
              data_dir='../BraTS_data',
              model_dir='./models',
              output_dir='./predictions',
              use_trained=False,
              fast_mode=False):
    """
    Run inference on a patient's MRI data
    
    Args:
        patient_id: Patient ID (e.g., 'BraTS2021_00451')
        data_dir: Directory containing input MRI data
        model_dir: Directory containing trained models
        output_dir: Directory to save predictions
        use_trained: Whether to use trained models (if False, uses untrained models)
        fast_mode: Use fast GLCM extraction (faster but less accurate)
    
    Returns:
        prediction: Segmentation mask (H, W, D)
    """
    
    print("\n" + "="*80)
    print(f"INFERENCE: {patient_id}")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Use trained models: {use_trained}")
    print(f"Fast mode: {fast_mode}")
    
    # Load input data
    print("\n[1] Loading input MRI data...")
    volume = load_mri_volume(data_dir, patient_id)
    
    # Initialize pipeline
    print("\n[2] Initializing segmentation pipeline...")
    pipeline = BrainTumorSegmentationPipeline(
        target_shape=(240, 240, 160, 4),
        glcm_window_size=5 if not fast_mode else 3,
        glcm_levels=16,
        vpt_n_features=10,
        vpt_method='variance',
        patch_size=(128, 128, 128),
        num_classes=4,
        ensemble_strategy='weighted',
        ensemble_alpha=0.4,
        ensemble_beta=0.6,
        device=device
    )
    
    # Load trained models if available
    if use_trained:
        model_dir = Path(model_dir)
        
        cnn_path = model_dir / 'cnn_best.pth'
        unet_path = model_dir / 'unet_best.pth'
        
        if cnn_path.exists():
            print(f"\n[3] Loading trained CNN model from {cnn_path}")
            pipeline.cnn_model.load_model(str(cnn_path))
        else:
            print(f"\n[WARNING] CNN model not found at {cnn_path}")
            print("[WARNING] Using untrained CNN model")
        
        if unet_path.exists():
            print(f"[4] Loading trained U-Net model from {unet_path}")
            pipeline.unet_model.load_model(str(unet_path))
        else:
            print(f"[WARNING] U-Net model not found at {unet_path}")
            print("[WARNING] Using untrained U-Net model")
    else:
        print("\n[3] Using untrained models (for testing pipeline only)")
    
    # Run inference
    print("\n[5] Running segmentation pipeline...")
    print("     This may take several minutes...")
    
    start_time = time.time()
    
    prediction = pipeline.predict(
        volume,
        fast_mode=fast_mode,
        glcm_stride=10 if fast_mode else 1,
        pred_stride=(128, 128, 128)
    )
    
    elapsed = time.time() - start_time
    print(f"\n[6] Inference completed in {elapsed/60:.1f} minutes")
    print(f"     Prediction shape: {prediction.shape}")
    print(f"     Unique labels: {np.unique(prediction)}")
    
    # Save prediction
    print("\n[7] Saving prediction...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy
    npy_path = output_dir / f"{patient_id}_prediction.npy"
    np.save(npy_path, prediction)
    print(f"     Numpy saved to: {npy_path}")
    
    # Save as NIfTI
    nifti_path = output_dir / f"{patient_id}_prediction.nii.gz"
    ref_path = Path(data_dir) / f"{patient_id}_t1.nii.gz"
    save_prediction(prediction, nifti_path, ref_path)
    
    # Statistics
    print("\n[8] Prediction statistics:")
    total_voxels = prediction.size
    labels = {0: 'Background', 1: 'Necrotic', 2: 'Edema', 4: 'Enhancing'}
    
    for label, name in labels.items():
        if label in np.unique(prediction):
            count = np.sum(prediction == label)
            percent = (count / total_voxels) * 100
            print(f"     {name:.<20} {count:>8} voxels ({percent:>5.2f}%)")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    
    return prediction


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on brain MRI data')
    parser.add_argument('--patient', type=str, required=True,
                        help='Patient ID (e.g., BraTS2021_00451)')
    parser.add_argument('--data', type=str, default='../BraTS_data',
                        help='Path to input data directory')
    parser.add_argument('--models', type=str, default='./models',
                        help='Path to trained models directory')
    parser.add_argument('--output', type=str, default='./predictions',
                        help='Path to output directory')
    parser.add_argument('--trained', action='store_true',
                        help='Use trained models (default: False)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast mode (default: False)')
    
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("#  BRAIN TUMOR SEGMENTATION - INFERENCE  #".center(80))
    print("#"*80)
    print(f"\nConfiguration:")
    print(f"  Patient ID: {args.patient}")
    print(f"  Data dir: {args.data}")
    print(f"  Models dir: {args.models}")
    print(f"  Output dir: {args.output}")
    print(f"  Use trained models: {args.trained}")
    print(f"  Fast mode: {args.fast}")
    
    # Run inference
    prediction = inference(
        patient_id=args.patient,
        data_dir=args.data,
        model_dir=args.models,
        output_dir=args.output,
        use_trained=args.trained,
        fast_mode=args.fast
    )
    
    print(f"\n[INFO] Inference finished successfully!")
    print(f"[INFO] Prediction saved to {args.output}/{args.patient}_prediction.nii.gz")
