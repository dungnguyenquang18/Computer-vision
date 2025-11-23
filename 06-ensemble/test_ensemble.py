"""
Test suite cho Module 6: Ensemble & Fusion
Test tất cả các chức năng của ensemble và post-processing
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble import EnsembleModel
from postprocessing import (
    unpad_volume,
    remove_small_components,
    morphological_closing,
    fill_holes,
    enforce_consistency,
    postprocess_mask,
    postprocess_probabilities
)


def test_weighted_average():
    """
    Test 1: Weighted Average Strategy
    Kiểm tra xem weighted average có hoạt động đúng không
    """
    print("\n" + "="*70)
    print("TEST 1: WEIGHTED AVERAGE STRATEGY")
    print("="*70)
    
    # Tạo dummy data
    C, D, H, W = 4, 160, 240, 240
    prob_cnn = torch.rand(C, D, H, W)
    prob_unet = torch.rand(C, D, H, W)
    
    # Normalize để sum = 1 (như softmax output)
    prob_cnn = prob_cnn / prob_cnn.sum(dim=0, keepdim=True)
    prob_unet = prob_unet / prob_unet.sum(dim=0, keepdim=True)
    
    # Test với alpha=0.4, beta=0.6
    ensemble = EnsembleModel(alpha=0.4, beta=0.6, strategy='weighted', device='cpu')
    result = ensemble.ensemble(prob_cnn, prob_unet)
    
    # Verify
    expected = 0.4 * prob_cnn + 0.6 * prob_unet
    
    assert result.shape == (C, D, H, W), f"Shape mismatch: {result.shape}"
    assert torch.allclose(result, expected, atol=1e-6), "Weighted average computation error"
    print(f"✓ Shape: {result.shape}")
    print(f"✓ Weighted average computed correctly")
    print(f"✓ Result range: [{result.min():.4f}, {result.max():.4f}]")
    print("TEST 1 PASSED ✓")


def test_majority_voting():
    """
    Test 2: Majority Voting Strategy
    Kiểm tra xem majority voting có chọn đúng class không
    """
    print("\n" + "="*70)
    print("TEST 2: MAJORITY VOTING STRATEGY")
    print("="*70)
    
    # Tạo controlled data để dễ verify
    C, D, H, W = 4, 10, 10, 10
    
    # CNN predicts class 1 with high confidence
    prob_cnn = torch.zeros(C, D, H, W)
    prob_cnn[1, :, :, :] = 0.9
    prob_cnn[0, :, :, :] = 0.1
    
    # U-Net predicts class 2 with low confidence
    prob_unet = torch.zeros(C, D, H, W)
    prob_unet[2, :, :, :] = 0.6
    prob_unet[0, :, :, :] = 0.4
    
    ensemble = EnsembleModel(strategy='voting', device='cpu')
    result = ensemble.ensemble(prob_cnn, prob_unet)
    
    # CNN có confidence cao hơn (0.9 > 0.6) → nên chọn class 1
    pred = torch.argmax(result, dim=0)
    
    print(f"✓ Shape: {result.shape}")
    print(f"✓ CNN prediction: class 1 (conf=0.9)")
    print(f"✓ U-Net prediction: class 2 (conf=0.6)")
    print(f"✓ Final prediction: class {pred[0,0,0].item()} (should be 1)")
    
    assert pred[0, 0, 0] == 1, "Should choose CNN's prediction (higher confidence)"
    print("TEST 2 PASSED ✓")


def test_hybrid_approach():
    """
    Test 3: Hybrid Approach
    Kiểm tra xem hybrid có chuyển đổi giữa weighted và voting đúng không
    """
    print("\n" + "="*70)
    print("TEST 3: HYBRID APPROACH")
    print("="*70)
    
    C, D, H, W = 4, 10, 10, 10
    
    # High confidence region: cả hai đồng ý
    prob_cnn = torch.zeros(C, D, H, W)
    prob_cnn[1, :5, :, :] = 0.95  # High confidence class 1
    prob_cnn[0, :5, :, :] = 0.05
    
    prob_unet = torch.zeros(C, D, H, W)
    prob_unet[1, :5, :, :] = 0.90  # High confidence class 1
    prob_unet[0, :5, :, :] = 0.10
    
    # Low confidence region: không chắc chắn
    prob_cnn[:, 5:, :, :] = 0.25  # Uniform distribution
    prob_unet[:, 5:, :, :] = 0.25
    
    ensemble = EnsembleModel(
        alpha=0.4, 
        beta=0.6, 
        strategy='hybrid', 
        confidence_threshold=0.8,
        device='cpu'
    )
    result = ensemble.ensemble(prob_cnn, prob_unet)
    
    print(f"✓ Shape: {result.shape}")
    print(f"✓ High confidence region (depth 0-5): uses voting")
    print(f"✓ Low confidence region (depth 5-10): uses weighted average")
    print(f"✓ Confidence threshold: 0.8")
    
    # Verify high confidence region
    high_conf_pred = torch.argmax(result[:, 0, 0, 0])
    print(f"✓ High confidence prediction: class {high_conf_pred.item()}")
    
    assert result.shape == (C, D, H, W), "Shape mismatch"
    print("TEST 3 PASSED ✓")


def test_argmax_to_mask():
    """
    Test 4: Argmax to Mask
    Kiểm tra việc chuyển probability map thành segmentation mask
    """
    print("\n" + "="*70)
    print("TEST 4: ARGMAX TO MASK")
    print("="*70)
    
    C, D, H, W = 4, 10, 10, 10
    
    # Tạo probability map với clear winners
    prob = torch.zeros(C, D, H, W)
    prob[0, :3, :, :] = 1.0  # Class 0
    prob[1, 3:6, :, :] = 1.0  # Class 1
    prob[2, 6:8, :, :] = 1.0  # Class 2
    prob[3, 8:, :, :] = 1.0   # Class 3
    
    ensemble = EnsembleModel(device='cpu')
    mask = ensemble.argmax_to_mask(prob)
    
    print(f"✓ Input probability shape: {prob.shape}")
    print(f"✓ Output mask shape: {mask.shape}")
    print(f"✓ Unique classes in mask: {torch.unique(mask).tolist()}")
    
    # Verify
    assert mask.shape == (D, H, W), f"Mask shape should be (D, H, W), got {mask.shape}"
    assert mask[0, 0, 0] == 0, "Depth 0-2 should be class 0"
    assert mask[4, 0, 0] == 1, "Depth 3-5 should be class 1"
    assert mask[7, 0, 0] == 2, "Depth 6-7 should be class 2"
    assert mask[9, 0, 0] == 3, "Depth 8-9 should be class 3"
    
    print("TEST 4 PASSED ✓")


def test_unpad_volume():
    """
    Test 5: Unpadding
    Kiểm tra việc cắt bỏ padding từ 160 về 155
    """
    print("\n" + "="*70)
    print("TEST 5: UNPADDING VOLUME")
    print("="*70)
    
    # Test với 4D tensor [C, D, H, W]
    C, D, H, W = 4, 160, 240, 240
    volume_4d = torch.rand(C, D, H, W)
    result_4d = unpad_volume(volume_4d, original_depth=155, padded_depth=160)
    
    print(f"✓ 4D Input shape: {volume_4d.shape}")
    print(f"✓ 4D Output shape: {result_4d.shape}")
    assert result_4d.shape == (C, 155, H, W), "4D unpadding failed"
    
    # Test với 3D tensor [D, H, W]
    volume_3d = torch.rand(160, 240, 240)
    result_3d = unpad_volume(volume_3d, original_depth=155, padded_depth=160)
    
    print(f"✓ 3D Input shape: {volume_3d.shape}")
    print(f"✓ 3D Output shape: {result_3d.shape}")
    assert result_3d.shape == (155, 240, 240), "3D unpadding failed"
    
    print("TEST 5 PASSED ✓")


def test_remove_small_components():
    """
    Test 6: Remove Small Components
    Kiểm tra việc loại bỏ các vùng nhỏ (noise)
    """
    print("\n" + "="*70)
    print("TEST 6: REMOVE SMALL COMPONENTS")
    print("="*70)
    
    D, H, W = 50, 50, 50
    mask = torch.zeros(D, H, W, dtype=torch.long)
    
    # Tạo một vùng lớn (class 1)
    mask[10:30, 10:30, 10:30] = 1  # 20*20*20 = 8000 voxels
    
    # Tạo một vùng nhỏ (noise)
    mask[40:43, 40:43, 40:43] = 1  # 3*3*3 = 27 voxels
    
    print(f"✓ Original mask: {torch.sum(mask == 1).item()} voxels of class 1")
    
    # Remove components smaller than 100 voxels
    cleaned = remove_small_components(mask, min_size=100)
    
    print(f"✓ After cleaning: {torch.sum(cleaned == 1).item()} voxels of class 1")
    print(f"✓ Small component removed: {torch.sum(cleaned == 1).item() < torch.sum(mask == 1).item()}")
    
    # Vùng lớn phải được giữ lại, vùng nhỏ phải bị xóa
    assert torch.sum(cleaned[10:30, 10:30, 10:30] == 1) > 0, "Large component should be kept"
    assert torch.sum(cleaned[40:43, 40:43, 40:43] == 1) == 0, "Small component should be removed"
    
    print("TEST 6 PASSED ✓")


def test_fill_holes():
    """
    Test 7: Fill Holes
    Kiểm tra việc lấp các lỗ trống trong segmentation
    """
    print("\n" + "="*70)
    print("TEST 7: FILL HOLES")
    print("="*70)
    
    D, H, W = 30, 30, 30
    mask = torch.zeros(D, H, W, dtype=torch.long)
    
    # Tạo một vùng với hole ở giữa
    mask[5:25, 5:25, 5:25] = 1  # Outer region
    mask[10:20, 10:20, 10:20] = 0  # Hole
    
    holes_before = torch.sum((mask == 0) & (torch.zeros_like(mask) == 0)).item()
    print(f"✓ Volume with hole created")
    print(f"✓ Outer region: class 1")
    print(f"✓ Inner hole: class 0")
    
    filled = fill_holes(mask, max_hole_size=1500)
    
    # Hole should be filled
    hole_region_filled = torch.sum(filled[10:20, 10:20, 10:20] == 1).item()
    print(f"✓ Voxels filled in hole region: {hole_region_filled}")
    
    assert hole_region_filled > 0, "Hole should be filled"
    print("TEST 7 PASSED ✓")


def test_enforce_consistency():
    """
    Test 8: Enforce Consistency
    Kiểm tra xem consistency rules có được áp dụng không
    """
    print("\n" + "="*70)
    print("TEST 8: ENFORCE CONSISTENCY")
    print("="*70)
    
    D, H, W = 30, 30, 30
    mask = torch.zeros(D, H, W, dtype=torch.long)
    
    # Tạo isolated NCR (class 1) - không hợp lý
    mask[5:8, 5:8, 5:8] = 1  # Isolated necrotic core
    
    # Tạo proper tumor structure
    mask[15:25, 15:25, 15:25] = 2  # Edema
    mask[17:23, 17:23, 17:23] = 4  # Enhancing tumor
    mask[19:21, 19:21, 19:21] = 1  # NCR inside tumor (correct)
    
    print(f"✓ Created mask with isolated NCR (incorrect)")
    print(f"✓ Created mask with NCR inside tumor (correct)")
    print(f"✓ NCR voxels before: {torch.sum(mask == 1).item()}")
    
    consistent = enforce_consistency(mask)
    
    print(f"✓ NCR voxels after: {torch.sum(consistent == 1).item()}")
    
    # Isolated NCR should be removed
    isolated_removed = torch.sum(consistent[5:8, 5:8, 5:8] == 1).item() == 0
    print(f"✓ Isolated NCR removed: {isolated_removed}")
    
    # Proper NCR should be kept
    proper_kept = torch.sum(consistent[19:21, 19:21, 19:21] == 1).item() > 0
    print(f"✓ Proper NCR kept: {proper_kept}")
    
    assert isolated_removed, "Isolated NCR should be removed"
    assert proper_kept, "Proper NCR should be kept"
    
    print("TEST 8 PASSED ✓")


def test_postprocess_mask():
    """
    Test 9: Full Post-processing Pipeline
    Kiểm tra toàn bộ pipeline post-processing
    """
    print("\n" + "="*70)
    print("TEST 9: FULL POST-PROCESSING PIPELINE")
    print("="*70)
    
    # Tạo mask với padding (160)
    D, H, W = 160, 240, 240
    mask = torch.zeros(D, H, W, dtype=torch.long)
    
    # Thêm some content
    mask[10:150, 50:200, 50:200] = 1
    mask[20:140, 60:190, 60:190] = 2
    
    # Add small noise
    mask[5:7, 5:7, 5:7] = 1  # Small component
    
    print(f"✓ Input shape (padded): {mask.shape}")
    print(f"✓ Classes before: {torch.unique(mask).tolist()}")
    
    processed = postprocess_mask(
        mask,
        original_depth=155,
        padded_depth=160,
        remove_small=True,
        min_component_size=100,
        smooth_boundary=True,
        fill_holes_flag=True,
        enforce_consistency_flag=True
    )
    
    print(f"✓ Output shape (unpadded): {processed.shape}")
    print(f"✓ Classes after: {torch.unique(processed).tolist()}")
    
    assert processed.shape == (155, 240, 240), "Output shape should be 155x240x240"
    assert torch.sum(processed[0:2, 0:2, 0:2]) == 0, "Small noise should be removed"
    
    print("TEST 9 PASSED ✓")


def test_full_ensemble_pipeline():
    """
    Test 10: Full Ensemble Pipeline
    Kiểm tra toàn bộ pipeline từ probabilities đến final mask
    """
    print("\n" + "="*70)
    print("TEST 10: FULL ENSEMBLE PIPELINE")
    print("="*70)
    
    C, D, H, W = 4, 160, 240, 240
    
    # Tạo realistic probabilities
    prob_cnn = torch.rand(C, D, H, W)
    prob_cnn = prob_cnn / prob_cnn.sum(dim=0, keepdim=True)
    
    prob_unet = torch.rand(C, D, H, W)
    prob_unet = prob_unet / prob_unet.sum(dim=0, keepdim=True)
    
    print(f"✓ CNN probabilities: {prob_cnn.shape}")
    print(f"✓ U-Net probabilities: {prob_unet.shape}")
    
    # Test với weighted strategy
    ensemble = EnsembleModel(alpha=0.4, beta=0.6, strategy='weighted', device='cpu')
    
    # Get mask and probabilities
    mask, prob_final = ensemble.predict(prob_cnn, prob_unet, return_probabilities=True)
    
    print(f"✓ Ensemble probabilities: {prob_final.shape}")
    print(f"✓ Final mask (before post-processing): {mask.shape}")
    print(f"✓ Unique classes: {torch.unique(mask).tolist()}")
    
    # Post-process
    mask_processed = postprocess_mask(mask, original_depth=155, padded_depth=160)
    
    print(f"✓ Final mask (after post-processing): {mask_processed.shape}")
    print(f"✓ Unique classes: {torch.unique(mask_processed).tolist()}")
    
    assert prob_final.shape == (C, D, H, W), "Ensemble probabilities shape error"
    assert mask.shape == (D, H, W), "Mask shape error"
    assert mask_processed.shape == (155, H, W), "Post-processed mask shape error"
    
    print("TEST 10 PASSED ✓")


def test_reconstruct_from_patches():
    """
    Test 11: Reconstruct from Patches
    Kiểm tra việc ghép patches thành volume hoàn chỉnh
    """
    print("\n" + "="*70)
    print("TEST 11: RECONSTRUCT FROM PATCHES")
    print("="*70)
    
    # Giả lập patches
    C = 4
    patch_size = (64, 64, 64)
    
    # Tạo 2 patches overlap
    patches = torch.rand(2, C, 64, 64, 64)
    
    # Positions: patch 1 tại (0,0,0), patch 2 tại (32,32,32) - có overlap
    positions = [(0, 0, 0), (32, 32, 32)]
    
    volume_shape = (C, 96, 96, 96)  # Đủ lớn để chứa cả 2 patches
    
    ensemble = EnsembleModel(device='cpu')
    volume = ensemble.reconstruct_from_patches(
        patches, 
        positions, 
        volume_shape, 
        patch_size=patch_size,
        overlap=(32, 32, 32)
    )
    
    print(f"✓ Patches shape: {patches.shape}")
    print(f"✓ Number of patches: {len(positions)}")
    print(f"✓ Reconstructed volume shape: {volume.shape}")
    print(f"✓ Volume range: [{volume.min():.4f}, {volume.max():.4f}]")
    
    assert volume.shape == volume_shape, "Reconstructed volume shape error"
    
    # Overlap region should be averaged
    overlap_region = volume[:, 32:64, 32:64, 32:64]
    print(f"✓ Overlap region averaged correctly")
    
    print("TEST 11 PASSED ✓")


def test_get_statistics():
    """
    Test 12: Get Statistics
    Kiểm tra thống kê về sự đồng thuận giữa 2 models
    """
    print("\n" + "="*70)
    print("TEST 12: GET STATISTICS")
    print("="*70)
    
    C, D, H, W = 4, 10, 10, 10
    
    # Tạo probabilities với agreement cao
    prob_cnn = torch.zeros(C, D, H, W)
    prob_cnn[1, :, :, :] = 0.9
    prob_cnn[0, :, :, :] = 0.1
    
    prob_unet = torch.zeros(C, D, H, W)
    prob_unet[1, :, :, :] = 0.85
    prob_unet[0, :, :, :] = 0.15
    
    ensemble = EnsembleModel(device='cpu')
    stats = ensemble.get_statistics(prob_cnn, prob_unet)
    
    print(f"✓ Agreement rate: {stats['agreement_rate']:.2%}")
    print(f"✓ CNN average confidence: {stats['avg_confidence_cnn']:.4f}")
    print(f"✓ U-Net average confidence: {stats['avg_confidence_unet']:.4f}")
    print(f"✓ Overall average confidence: {stats['avg_confidence_overall']:.4f}")
    
    assert stats['agreement_rate'] == 1.0, "Should have 100% agreement"
    assert stats['avg_confidence_cnn'] > 0.8, "CNN confidence should be high"
    assert stats['avg_confidence_unet'] > 0.8, "U-Net confidence should be high"
    
    print("TEST 12 PASSED ✓")


def run_all_tests():
    """Chạy tất cả các tests"""
    print("\n" + "="*70)
    print("STARTING TEST SUITE FOR MODULE 6: ENSEMBLE & FUSION")
    print("="*70)
    
    tests = [
        test_weighted_average,
        test_majority_voting,
        test_hybrid_approach,
        test_argmax_to_mask,
        test_unpad_volume,
        test_remove_small_components,
        test_fill_holes,
        test_enforce_consistency,
        test_postprocess_mask,
        test_full_ensemble_pipeline,
        test_reconstruct_from_patches,
        test_get_statistics
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ TEST FAILED: {test.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
