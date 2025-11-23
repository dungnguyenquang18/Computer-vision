# coding: utf-8
"""
Training script for Brain Tumor Segmentation models
Train CNN3D and UNet3D models on BraTS dataset
"""

import sys
import numpy as np
import torch
from pathlib import Path
import time
from datetime import datetime

# Add module paths
sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).parent / "glcm_extraction"))
sys.path.insert(0, str(Path(__file__).parent / "vpt_selection"))
sys.path.insert(0, str(Path(__file__).parent / "cnn3d"))
sys.path.insert(0, str(Path(__file__).parent / "unet3d"))

from preprocessing import Preprocessor
from glcm_extraction import GLCMExtractorTorch
from vpt_selection import VPTSelector
from cnn3d import CNN3DModel
from unet3d import UNet3DModel

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("[ERROR] nibabel not installed. Install: pip install nibabel")
    sys.exit(1)


class BraTSDataLoader:
    """Data loader for BraTS dataset"""
    
    def __init__(self, data_dir, patch_size=(128, 128, 128), num_patches_per_volume=4, device='cuda'):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_patches_per_volume = num_patches_per_volume
        self.device = device
        
        # Feature extraction (GLCM with GPU support)
        self.preprocessor = Preprocessor(target_shape=(240, 240, 160, 4))
        self.glcm_extractor = GLCMExtractorTorch(window_size=3, levels=16, device=device)
        self.vpt_selector = VPTSelector(n_features=8, selection_method='variance')
        
        print(f"[DataLoader] Initialized")
        print(f"  Data dir: {data_dir}")
        print(f"  Patch size: {patch_size}")
        print(f"  Patches per volume: {num_patches_per_volume}")
    
    def load_patient(self, patient_id):
        """Load one patient's data"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        
        # Load modalities
        volume_list = []
        for mod in modalities:
            filepath = self.data_dir / f"{patient_id}_{mod}.nii.gz"
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            nifti = nib.load(str(filepath))
            data = nifti.get_fdata()
            volume_list.append(data)
        
        volume = np.stack(volume_list, axis=-1).astype(np.float32)
        
        # Load segmentation
        seg_file = self.data_dir / f"{patient_id}_seg.nii.gz"
        if seg_file.exists():
            nifti = nib.load(str(seg_file))
            seg = nifti.get_fdata().astype(np.uint8)
        else:
            seg = None
        
        return volume, seg
    
    def extract_features(self, volume):
        """Extract features from volume"""
        # Preprocess
        processed = self.preprocessor.preprocess(volume)
        
        # GLCM (DON'T use fast mode for training to preserve dimensions)
        print("\n[DataLoader] Extracting GLCM features (full mode)...")
        glcm_features = self.glcm_extractor.extract_features(processed)
        print(f"  GLCM shape: {glcm_features.shape}")
        
        # VPT selection
        selected = self.vpt_selector.select_features(glcm_features)
        print(f"  Selected features shape: {selected.shape}")
        
        return selected
    
    def extract_random_patches(self, volume, segmentation, num_patches):
        """Extract random patches from volume"""
        H, W, D, C = volume.shape
        pH, pW, pD = self.patch_size
        
        # Check if patch extraction is possible
        if H < pH or W < pW or D < pD:
            print(f"      [WARNING] Volume {(H,W,D)} too small for patch {self.patch_size}")
            print(f"      Using smaller patch size: {(min(H,pH), min(W,pW), min(D,pD))}")
            pH = min(H, pH)
            pW = min(W, pW)
            pD = min(D, pD)
        
        patches_x = []
        patches_y = []
        
        for _ in range(num_patches):
            # Random location (ensure patch fits)
            h = np.random.randint(0, max(1, H - pH + 1))
            w = np.random.randint(0, max(1, W - pW + 1))
            d = np.random.randint(0, max(1, D - pD + 1))
            
            # Extract patch
            patch_x = volume[h:h+pH, w:w+pW, d:d+pD, :]
            patch_y = segmentation[h:h+pH, w:w+pW, d:d+pD]
            
            # Pad if needed to match target patch size
            if patch_x.shape[:3] != self.patch_size:
                pad_h = self.patch_size[0] - patch_x.shape[0]
                pad_w = self.patch_size[1] - patch_x.shape[1]
                pad_d = self.patch_size[2] - patch_x.shape[2]
                patch_x = np.pad(patch_x, ((0,pad_h), (0,pad_w), (0,pad_d), (0,0)))
                patch_y = np.pad(patch_y, ((0,pad_h), (0,pad_w), (0,pad_d)))
            
            patches_x.append(patch_x)
            patches_y.append(patch_y)
        
        if len(patches_x) == 0:
            raise ValueError("No patches extracted!")
        
        return np.array(patches_x), np.array(patches_y)
    
    def prepare_training_data(self, patient_ids):
        """Prepare training data from patient IDs"""
        all_patches_x = []
        all_patches_y = []
        
        print(f"\n[DataLoader] Preparing training data from {len(patient_ids)} patients...")
        
        for i, patient_id in enumerate(patient_ids):
            print(f"  [{i+1}/{len(patient_ids)}] Processing {patient_id}...")
            
            try:
                # Load patient
                volume, seg = self.load_patient(patient_id)
                
                # Extract features
                features = self.extract_features(volume)
                
                # Extract patches
                patches_x, patches_y = self.extract_random_patches(
                    features, seg, self.num_patches_per_volume
                )
                
                all_patches_x.append(patches_x)
                all_patches_y.append(patches_y)
                
                print(f"      Extracted {len(patches_x)} patches")
                
            except Exception as e:
                print(f"      [ERROR] Failed to process {patient_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Check if any patches were extracted
        if len(all_patches_x) == 0:
            raise ValueError("No training data! All patients failed to process.")
        
        # Concatenate all patches
        X = np.concatenate(all_patches_x, axis=0)
        y = np.concatenate(all_patches_y, axis=0)
        
        print(f"\n[DataLoader] Data preparation complete!")
        print(f"  Total patches: {len(X)}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        return X, y


def train_model(model_type='cnn', data_dir='../BraTS_data', 
                epochs=50, batch_size=2, save_dir='./models'):
    """
    Train CNN3D or UNet3D model
    
    Args:
        model_type: 'cnn' or 'unet'
        data_dir: Directory containing BraTS data
        epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save trained models
    """
    
    print("\n" + "="*80)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("="*80)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Data loader
    print("\n[1] Setting up data loader...")
    loader = BraTSDataLoader(
        data_dir=data_dir,
        patch_size=(128, 128, 128),
        num_patches_per_volume=4,
        device=device
    )
    
    # For demo: use single patient (in practice, use multiple patients)
    patient_ids = ["BraTS2021_00451"]
    
    print("\n[2] Preparing training data...")
    X_train, y_train = loader.prepare_training_data(patient_ids)
    
    # Convert labels to one-hot for CNN, keep as is for UNet
    if model_type == 'cnn':
        # One-hot encode for CNN
        num_classes = 4
        y_train_onehot = np.zeros((len(y_train), num_classes), dtype=np.float32)
        for i in range(len(y_train)):
            # Take most common label in patch as class
            unique, counts = np.unique(y_train[i], return_counts=True)
            most_common = unique[np.argmax(counts)]
            # Map label 4 to index 3
            class_idx = most_common if most_common < 4 else 3
            y_train_onehot[i, class_idx] = 1.0
        y_train_processed = y_train_onehot
    else:
        # Keep as segmentation masks for UNet
        y_train_processed = y_train
    
    # Split train/val (80/20)
    split_idx = int(0.8 * len(X_train))
    X_val = X_train[split_idx:]
    y_val = y_train_processed[split_idx:]
    X_train = X_train[:split_idx]
    y_train_processed = y_train_processed[:split_idx]
    
    print(f"\n[3] Data split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    
    # Initialize model
    print(f"\n[4] Initializing {model_type.upper()} model...")
    input_shape = (*loader.patch_size, loader.vpt_selector.n_features)
    
    if model_type == 'cnn':
        model = CNN3DModel(input_shape=input_shape, num_classes=4)
    else:
        model = UNet3DModel(input_shape=input_shape, num_classes=4, base_filters=16)
    
    model.summary()
    
    # Train
    print(f"\n[5] Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    start_time = time.time()
    
    history = model.train(
        train_data=(X_train, y_train_processed),
        val_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        save_path=str(save_dir / f"{model_type}_best.pth")
    )
    
    elapsed = time.time() - start_time
    print(f"\n[6] Training completed in {elapsed/60:.1f} minutes")
    
    # Save final model
    final_path = save_dir / f"{model_type}_final.pth"
    model.save_model(str(final_path))
    print(f"[7] Final model saved to: {final_path}")
    
    # Save training history
    history_path = save_dir / f"{model_type}_history.npy"
    np.save(history_path, history)
    print(f"[8] Training history saved to: {history_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train brain tumor segmentation model')
    parser.add_argument('--model', type=str, default='unet', choices=['cnn', 'unet'],
                        help='Model type: cnn or unet (default: unet)')
    parser.add_argument('--data', type=str, default='../BraTS_data',
                        help='Path to BraTS data directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--save', type=str, default='./models',
                        help='Directory to save models (default: ./models)')
    
    args = parser.parse_args()
    
    print("\n" + "#"*80)
    print("#  BRAIN TUMOR SEGMENTATION - TRAINING SCRIPT  #".center(80))
    print("#"*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Data dir: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Save dir: {args.save}")
    
    # Train model
    model, history = train_model(
        model_type=args.model,
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        save_dir=args.save
    )
    
    print("\n[INFO] Training finished successfully!")
    print(f"[INFO] Load trained model for inference using inference.py")
