# coding: utf-8
"""
Simple test for brain tumor segmentation pipeline - ASCII only version
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import pipeline
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import BrainTumorSegmentationPipeline


def test_small_pipeline():
    """Test với volume nhỏ"""
    print("\n" + "="*80)
    print("TESTING BRAIN TUMOR SEGMENTATION PIPELINE")
    print("="*80)
    
    # Create test volume (50x50x16x4)
    print("\n[1] Creating test volume (50x50x16x4)...")
    volume = np.random.randn(50, 50, 16, 4).astype(np.float32)
    volume[15:25, 15:25, 5:10, :] += 3.0  # Bright region
    volume[30:40, 30:40, 8:12, :] -= 2.0  # Dark region
    print(f"    Volume shape: {volume.shape}")
    print(f"    Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Initialize pipeline
    print("\n[2] Initializing pipeline...")
    pipeline = BrainTumorSegmentationPipeline(
        target_shape=(50, 50, 16, 4),
        glcm_window_size=3,
        glcm_levels=8,
        vpt_n_features=5,
        vpt_method='variance',
        patch_size=(64, 64, 16),
        num_classes=4,
        ensemble_strategy='weighted',
        device='cpu'
    )
    
    pipeline.get_pipeline_summary()
    
    # Test individual steps
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL STEPS")
    print("="*80)
    
    try:
        # Step 1: Preprocessing
        print("\n[3] Step 1: Preprocessing...")
        preprocessed = pipeline.preprocess_step(volume)
        assert preprocessed.shape == (50, 50, 16, 4)
        print(f"    [OK] Shape: {preprocessed.shape}")
        
        # Step 2: GLCM (fast mode)
        print("\n[4] Step 2: GLCM Extraction (fast mode)...")
        glcm_features = pipeline.extract_glcm_features_step(
            preprocessed, fast_mode=True, stride=15
        )
        print(f"    [OK] Shape: {glcm_features.shape}")
        print(f"    [OK] Range: [{glcm_features.min():.4f}, {glcm_features.max():.4f}]")
        
        # Step 3: VPT Selection
        print("\n[5] Step 3: VPT Feature Selection...")
        selected_features = pipeline.select_features_step(glcm_features)
        assert selected_features.shape[-1] == 5
        print(f"    [OK] Shape: {selected_features.shape}")
        
        # Step 4 & 5: CNN and U-Net (test with single patch only)
        print("\n[6] Step 4 & 5: Testing CNN and U-Net models...")
        print("    NOTE: Models are untrained, predictions are random")
        
        # We need to create a proper-sized patch (64x64x16x5)
        # But our features from fast GLCM are smaller, so pad them
        h, w, d, c = selected_features.shape
        test_patch = np.zeros((64, 64, 16, 5), dtype=np.float32)
        test_patch[:min(h,64), :min(w,64), :min(d,16), :] = selected_features[:64, :64, :16, :]
        
        print(f"    Testing with patch shape: {test_patch.shape}")
        
        # CNN prediction
        prob_cnn = pipeline.cnn_model.predict_patch(test_patch)
        print(f"    [OK] CNN output shape: {prob_cnn.shape}")
        
        # U-Net prediction  
        prob_unet = pipeline.unet_model.predict_patch(test_patch)
        print(f"    [OK] U-Net output shape: {prob_unet.shape}")
        
        # Step 6: Ensemble (with dummy full-size probabilities)
        print("\n[7] Step 6: Ensemble...")
        H, W, D = 50, 50, 16
        prob_cnn_full = np.random.rand(H, W, D, 4).astype(np.float32)
        prob_unet_full = np.random.rand(H, W, D, 4).astype(np.float32)
        prob_cnn_full = prob_cnn_full / prob_cnn_full.sum(axis=-1, keepdims=True)
        prob_unet_full = prob_unet_full / prob_unet_full.sum(axis=-1, keepdims=True)
        
        mask = pipeline.ensemble_step(prob_cnn_full, prob_unet_full)
        print(f"    [OK] Mask shape: {mask.shape}")
        print(f"    [OK] Unique labels: {np.unique(mask)}")
        
        # Step 7: Post-processing
        print("\n[8] Step 7: Post-processing...")
        mask_final = pipeline.postprocess_step(mask)
        print(f"    [OK] Final shape: {mask_final.shape}")
        
        print("\n" + "="*80)
        print("[OK] ALL INDIVIDUAL STEPS PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_summary():
    """Test model architectures"""
    print("\n" + "="*80)
    print("TESTING MODEL ARCHITECTURES")
    print("="*80)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "cnn3d"))
        sys.path.insert(0, str(Path(__file__).parent / "unet3d"))
        from cnn3d_model import CNN3DModel
        from unet3d_model import UNet3DModel
        
        print("\n[1] CNN3D Model...")
        cnn = CNN3DModel(input_shape=(64, 64, 16, 5), num_classes=4)
        cnn.summary()
        print("    [OK] CNN3D architecture created")
        
        print("\n[2] U-Net3D Model...")
        unet = UNet3DModel(input_shape=(64, 64, 16, 5), num_classes=4, base_filters=16)
        unet.summary()
        print("    [OK] U-Net3D architecture created")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("#  BRAIN TUMOR SEGMENTATION PIPELINE - SIMPLE TEST  #".center(80))
    print("#"*80)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    results = {}
    
    print("\n>>> Test 1: Model Architectures")
    results['models'] = test_model_summary()
    
    print("\n>>> Test 2: Pipeline with Small Volume")
    results['pipeline'] = test_small_pipeline()
    
    # Summary
    print("\n" + "#"*80)
    print("#  TEST SUMMARY  #".center(80))
    print("#"*80)
    
    for name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {name.upper():.<40} {status}")
    
    if all(results.values()):
        print("\n" + "="*80)
        print("  ALL TESTS PASSED!".center(80))
        print("="*80)
    else:
        print("\n" + "="*80)
        print("  SOME TESTS FAILED!".center(80))
        print("="*80)

