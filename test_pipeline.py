"""
Test Case Nhỏ: Kiểm tra Full Pipeline với dữ liệu synthetic nhỏ
Test từng bước và full pipeline với volume nhỏ để debug
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Import pipeline
sys.path.insert(0, str(Path(__file__).parent))
from full_pipeline import BrainTumorSegmentationPipeline


def test_pipeline_with_small_volume():
    """
    Test pipeline với volume nhỏ (50x50x16x4) để nhanh
    """
    print("\n" + "="*80)
    print("TEST CASE: Small Volume Pipeline Test")
    print("="*80)
    
    # Tạo synthetic volume nhỏ (50x50x16x4)
    # Giả lập 4 kênh MRI: T1, T1ce, T2, FLAIR
    print("\n[1] Creating synthetic test volume...")
    H, W, D, C = 50, 50, 16, 4
    volume = np.random.randn(H, W, D, C).astype(np.float32)
    
    # Thêm một vài "tumor-like" regions để test
    # Region 1: Bright spot (giả lập enhancing tumor)
    volume[15:25, 15:25, 5:10, :] += 3.0
    # Region 2: Dark spot (giả lập necrotic core)
    volume[30:40, 30:40, 8:12, :] -= 2.0
    
    print(f"   Volume shape: {volume.shape}")
    print(f"   Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    # Khởi tạo pipeline với config nhỏ gọn
    # Note: patch_size phải đủ lớn để qua được 3 lần pooling (/2 mỗi lần = /8)
    # Min size: 8*2*2*2 = 64 cho mỗi chiều
    print("\n[2] Initializing pipeline with small config...")
    pipeline = BrainTumorSegmentationPipeline(
        target_shape=(50, 50, 16, 4),  # Giữ nguyên size nhỏ
        glcm_window_size=3,            # Window nhỏ
        glcm_levels=8,                 # Ít mức xám hơn
        vpt_n_features=5,              # Chỉ giữ 5 features
        vpt_method='variance',
        patch_size=(64, 64, 16),       # Patch đủ lớn (64x64x16) để qua pooling
        num_classes=4,
        ensemble_strategy='weighted',
        ensemble_alpha=0.4,
        ensemble_beta=0.6,
        device='cpu'                   # Force CPU để test
    )
    
    # In summary
    pipeline.get_pipeline_summary()
    
    # Test từng bước riêng lẻ
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL STEPS")
    print("="*80)
    
    try:
        # Step 1: Preprocessing
        print("\n[3] Testing Preprocessing...")
        preprocessed = pipeline.preprocess_step(volume)
        assert preprocessed.shape == (50, 50, 16, 4), f"Wrong shape: {preprocessed.shape}"
        print(f"   ✓ Preprocessing OK - Shape: {preprocessed.shape}")
        
        # Step 2: GLCM (fast mode with large stride)
        print("\n[4] Testing GLCM Extraction (fast mode)...")
        glcm_features = pipeline.extract_glcm_features_step(
            preprocessed, 
            fast_mode=True, 
            stride=10  # Large stride for speed
        )
        print(f"   ✓ GLCM Extraction OK - Shape: {glcm_features.shape}")
        print(f"   ✓ Features range: [{glcm_features.min():.4f}, {glcm_features.max():.4f}]")
        
        # Step 3: VPT Selection
        print("\n[5] Testing VPT Feature Selection...")
        selected_features = pipeline.select_features_step(glcm_features)
        assert selected_features.shape[-1] == 5, f"Wrong features: {selected_features.shape}"
        print(f"   ✓ VPT Selection OK - Shape: {selected_features.shape}")
        
        # Step 4: CNN Prediction (chỉ test model architecture, không train)
        print("\n[6] Testing CNN Model (inference only, untrained)...")
        print("   NOTE: Models are untrained, predictions will be random")
        
        # Extract one patch to test (must match patch_size = 64x64x16)
        # But selected_features is only (5,5,16,5) from fast GLCM, so we pad it
        print(f"   Selected features shape: {selected_features.shape}")
        
        # Create a properly sized patch for testing
        test_patch = np.zeros((64, 64, 16, 5), dtype=np.float32)
        h, w, d, c = selected_features.shape
        test_patch[:h, :w, :d, :] = selected_features
        
        prob_patch_cnn = pipeline.cnn_model.predict_patch(test_patch)
        print(f"   ✓ CNN forward pass OK")
        print(f"   ✓ Patch input: {test_patch.shape}")
        print(f"   ✓ Patch output: {prob_patch_cnn.shape}")
        print(f"   ✓ Probabilities sum: {prob_patch_cnn.sum():.4f}")
        
        # Step 5: U-Net Prediction
        print("\n[7] Testing U-Net Model (inference only, untrained)...")
        prob_patch_unet = pipeline.unet_model.predict_patch(test_patch)
        print(f"   ✓ U-Net forward pass OK")
        print(f"   ✓ Patch output: {prob_patch_unet.shape}")
        
        # Create dummy probability maps for ensemble test
        print("\n[8] Testing Ensemble (with dummy data)...")
        H, W, D = 50, 50, 16
        prob_cnn_dummy = np.random.rand(H, W, D, 4).astype(np.float32)
        prob_unet_dummy = np.random.rand(H, W, D, 4).astype(np.float32)
        
        # Normalize to valid probabilities
        prob_cnn_dummy = prob_cnn_dummy / prob_cnn_dummy.sum(axis=-1, keepdims=True)
        prob_unet_dummy = prob_unet_dummy / prob_unet_dummy.sum(axis=-1, keepdims=True)
        
        mask = pipeline.ensemble_step(prob_cnn_dummy, prob_unet_dummy)
        print(f"   ✓ Ensemble OK - Mask shape: {mask.shape}")
        print(f"   ✓ Unique labels: {np.unique(mask)}")
        
        # Test post-processing
        print("\n[9] Testing Post-processing...")
        # Create dummy mask with 16 slices
        mask_16 = np.random.randint(0, 4, size=(50, 50, 16), dtype=np.uint8)
        mask_15 = pipeline.postprocess_step(mask_16)
        print(f"   ✓ Post-processing OK - Shape: {mask_15.shape}")
        
        print("\n" + "="*80)
        print("✓ ALL INDIVIDUAL STEPS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ ERROR in individual steps: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full pipeline (warning: sẽ chậm hơn)
    print("\n" + "="*80)
    print("TESTING FULL PIPELINE (END-TO-END)")
    print("="*80)
    print("\nWARNING: This will take some time...")
    print("Press Ctrl+C to skip full pipeline test\n")
    
    try:
        # Run full pipeline với stride lớn để nhanh
        final_mask = pipeline.predict(
            volume,
            fast_mode=True,      # GLCM fast mode
            glcm_stride=15,      # Very large stride for GLCM
            pred_stride=(64, 64, 16)  # No overlap for CNN/UNet (must match patch_size)
        )
        
        print("\n" + "="*80)
        print("✓ FULL PIPELINE TEST PASSED!")
        print("="*80)
        print(f"Final mask shape: {final_mask.shape}")
        print(f"Unique labels: {np.unique(final_mask)}")
        
        # Basic statistics
        total_voxels = final_mask.size
        for label in np.unique(final_mask):
            count = np.sum(final_mask == label)
            percentage = (count / total_voxels) * 100
            label_name = {0: 'Background', 1: 'Necrotic', 2: 'Edema', 3: 'Enhancing', 4: 'Enhancing'}
            print(f"  Label {label} ({label_name.get(label, 'Unknown')}): {count} voxels ({percentage:.2f}%)")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Full pipeline test skipped by user")
        return True
    except Exception as e:
        print(f"\n✗ ERROR in full pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_architectures():
    """
    Test model architectures với dummy data
    """
    print("\n" + "="*80)
    print("TEST CASE: Model Architecture Test")
    print("="*80)
    
    try:
        # Add module paths
        sys.path.insert(0, str(Path(__file__).parent / "cnn3d"))
        sys.path.insert(0, str(Path(__file__).parent / "unet3d"))
        from cnn3d_model import CNN3DModel
        from unet3d_model import UNet3DModel
        
        print("\n[1] Testing CNN3D architecture...")
        # Use larger patch size (64x64x16) to avoid pooling size issues
        cnn = CNN3DModel(input_shape=(64, 64, 16, 5), num_classes=4)
        cnn.summary()
        print("   ✓ CNN3D architecture OK")
        
        print("\n[2] Testing UNet3D architecture...")
        unet = UNet3DModel(input_shape=(64, 64, 16, 5), num_classes=4, base_filters=16)
        unet.summary()
        print("   ✓ UNet3D architecture OK")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR in model architectures: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    """
    Test CUDA vs CPU compatibility
    """
    print("\n" + "="*80)
    print("TEST CASE: Device Compatibility Test")
    print("="*80)
    
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Test tensor operations
    try:
        # CPU test
        x_cpu = torch.randn(2, 4, 8, 8, 8)
        print(f"\n✓ CPU tensor creation OK: {x_cpu.shape}")
        
        # GPU test if available
        if cuda_available:
            x_gpu = x_cpu.cuda()
            print(f"✓ GPU tensor creation OK: {x_gpu.shape}, device: {x_gpu.device}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR in device test: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#  BRAIN TUMOR SEGMENTATION PIPELINE - TEST SUITE  #".center(80))
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Run all tests
    results = {}
    
    print("\n\n>>> Running Test 1: Device Compatibility")
    results['device'] = test_device_compatibility()
    
    print("\n\n>>> Running Test 2: Model Architectures")
    results['models'] = test_model_architectures()
    
    print("\n\n>>> Running Test 3: Full Pipeline (Small Volume)")
    results['pipeline'] = test_pipeline_with_small_volume()
    
    # Summary
    print("\n\n" + "#"*80)
    print("#  TEST SUMMARY  #".center(80))
    print("#"*80)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.upper():.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("  ALL TESTS PASSED! ✓".center(80))
        print("="*80)
    else:
        print("\n" + "="*80)
        print("  SOME TESTS FAILED! ✗".center(80))
        print("="*80)
    
    print("\n")
