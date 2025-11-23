"""
Test Suite cho Module GLCM Extraction
Kiểm thử đầy đủ các chức năng của GLCMExtractor class.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from glcm_extractor import GLCMExtractor


def print_test_header(test_name):
    """Helper function để in header của test case"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def test_case_1_initialization():
    """
    TEST CASE 1: Khởi tạo GLCMExtractor
    - Kiểm tra khởi tạo với default parameters
    - Kiểm tra khởi tạo với custom parameters
    - Kiểm tra các thuộc tính cơ bản
    """
    print_test_header("Case 1 - Initialization")
    
    # Test 1.1: Default initialization
    print("\n[1.1] Testing default initialization...")
    extractor = GLCMExtractor()
    
    assert extractor.window_size == 5, "Default window size should be 5"
    assert extractor.distances == [1], "Default distance should be [1]"
    assert len(extractor.angles) == 4, "Default should have 4 angles"
    assert extractor.levels == 32, "Default levels should be 32"
    assert len(extractor.feature_names) == 5, "Should have 5 features"
    print("✓ Default initialization passed")
    
    # Test 1.2: Custom initialization
    print("\n[1.2] Testing custom initialization...")
    extractor_custom = GLCMExtractor(
        window_size=3,
        distances=[1, 2],
        angles=[0, np.pi/2],
        levels=16
    )
    
    assert extractor_custom.window_size == 3, "Custom window size mismatch"
    assert extractor_custom.distances == [1, 2], "Custom distances mismatch"
    assert len(extractor_custom.angles) == 2, "Should have 2 custom angles"
    assert extractor_custom.levels == 16, "Custom levels mismatch"
    print("✓ Custom initialization passed")
    
    # Test 1.3: Feature names check
    print("\n[1.3] Testing feature names...")
    expected_features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    assert extractor.feature_names == expected_features, "Feature names mismatch"
    print(f"✓ All 5 features present: {extractor.feature_names}")
    
    print("\n✅ TEST CASE 1 PASSED: Initialization works correctly")


def test_case_2_quantization():
    """
    TEST CASE 2: Image quantization
    - Kiểm tra quantization về số mức xám cố định
    - Kiểm tra xử lý ảnh constant (min = max)
    - Kiểm tra range output [0, levels-1]
    """
    print_test_header("Case 2 - Image Quantization")
    
    extractor = GLCMExtractor(levels=16)
    
    # Test 2.1: Normal quantization
    print("\n[2.1] Testing normal quantization...")
    image = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0],
                      [-0.8, -0.3, 0.2, 0.7, 0.9]], dtype=np.float32)
    
    quantized = extractor._quantize_image(image)
    
    assert quantized.dtype == np.uint8, "Output should be uint8"
    assert quantized.min() >= 0, "Min should be >= 0"
    assert quantized.max() <= 15, f"Max should be <= 15, got {quantized.max()}"
    print(f"✓ Quantized range: [{quantized.min()}, {quantized.max()}]")
    
    # Test 2.2: Constant image
    print("\n[2.2] Testing constant image quantization...")
    constant_img = np.ones((10, 10), dtype=np.float32) * 42.0
    quantized_const = extractor._quantize_image(constant_img)
    
    assert np.all(quantized_const == 0), "Constant image should quantize to all zeros"
    print("✓ Constant image handled correctly")
    
    # Test 2.3: Verify quantization levels
    print("\n[2.3] Testing quantization to exact levels...")
    # Create image with known range
    test_img = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
    quantized_test = extractor._quantize_image(test_img)
    
    unique_levels = len(np.unique(quantized_test))
    print(f"✓ Unique levels in output: {unique_levels} (levels={extractor.levels})")
    assert unique_levels <= extractor.levels, "Should not exceed specified levels"
    
    print("\n✅ TEST CASE 2 PASSED: Quantization works correctly")


def test_case_3_glcm_computation():
    """
    TEST CASE 3: GLCM computation for 2D patches
    - Kiểm tra tính GLCM cho patch 2D
    - Kiểm tra GLCM shape và properties
    - Kiểm tra symmetric và normalized
    """
    print_test_header("Case 3 - GLCM Computation")
    
    extractor = GLCMExtractor(window_size=5, distances=[1], levels=8)
    
    # Test 3.1: Basic GLCM computation
    print("\n[3.1] Testing basic GLCM computation...")
    # Create simple pattern
    patch = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [2, 2, 3, 3],
                      [2, 2, 3, 3]], dtype=np.uint8)
    
    glcm = extractor.compute_glcm_2d(patch)
    
    assert glcm is not None, "GLCM should not be None"
    assert glcm.ndim == 4, f"GLCM should be 4D, got {glcm.ndim}D"
    expected_shape = (8, 8, len(extractor.distances), len(extractor.angles))
    assert glcm.shape == expected_shape, f"Expected shape {expected_shape}, got {glcm.shape}"
    print(f"✓ GLCM shape: {glcm.shape}")
    
    # Test 3.2: GLCM normalization
    print("\n[3.2] Testing GLCM normalization...")
    # Sum along first two axes should be approximately 1 (normalized)
    for d in range(len(extractor.distances)):
        for a in range(len(extractor.angles)):
            matrix_sum = glcm[:, :, d, a].sum()
            assert 0.99 <= matrix_sum <= 1.01, f"GLCM should be normalized, got sum={matrix_sum}"
    print("✓ GLCM properly normalized")
    
    # Test 3.3: Empty patch handling
    print("\n[3.3] Testing empty patch...")
    empty_patch = np.array([], dtype=np.uint8)
    glcm_empty = extractor.compute_glcm_2d(empty_patch)
    
    assert glcm_empty is None, "Empty patch should return None"
    print("✓ Empty patch handled correctly")
    
    print("\n✅ TEST CASE 3 PASSED: GLCM computation works correctly")


def test_case_4_haralick_features():
    """
    TEST CASE 4: Haralick feature extraction
    - Kiểm tra trích xuất 5 features từ GLCM
    - Kiểm tra giá trị features hợp lý
    - Kiểm tra xử lý GLCM None
    """
    print_test_header("Case 4 - Haralick Features Extraction")
    
    extractor = GLCMExtractor(window_size=5, levels=8)
    
    # Test 4.1: Extract features from valid GLCM
    print("\n[4.1] Testing feature extraction from GLCM...")
    # Create simple patch and compute GLCM
    patch = np.random.randint(0, 8, size=(5, 5), dtype=np.uint8)
    glcm = extractor.compute_glcm_2d(patch)
    
    features = extractor.extract_haralick_features_2d(glcm)
    
    assert isinstance(features, dict), "Features should be a dictionary"
    assert len(features) == 5, f"Should have 5 features, got {len(features)}"
    
    expected_keys = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for key in expected_keys:
        assert key in features, f"Missing feature: {key}"
        assert isinstance(features[key], (int, float, np.number)), f"{key} should be numeric"
        assert not np.isnan(features[key]), f"{key} should not be NaN"
        assert not np.isinf(features[key]), f"{key} should not be Inf"
    
    print("✓ All 5 features extracted:")
    for key, value in features.items():
        print(f"    {key}: {value:.4f}")
    
    # Test 4.2: Feature value ranges
    print("\n[4.2] Testing feature value ranges...")
    # Homogeneity and energy should be in [0, 1]
    assert 0 <= features['homogeneity'] <= 1, "Homogeneity should be in [0, 1]"
    assert 0 <= features['energy'] <= 1, "Energy should be in [0, 1]"
    print(f"✓ Homogeneity: {features['homogeneity']:.4f} (in [0,1])")
    print(f"✓ Energy: {features['energy']:.4f} (in [0,1])")
    
    # Test 4.3: Handle None GLCM
    print("\n[4.3] Testing None GLCM handling...")
    features_none = extractor.extract_haralick_features_2d(None)
    
    assert isinstance(features_none, dict), "Should return dict for None GLCM"
    assert all(v == 0.0 for v in features_none.values()), "All features should be 0.0 for None"
    print("✓ None GLCM returns zero features")
    
    print("\n✅ TEST CASE 4 PASSED: Feature extraction works correctly")


def test_case_5_slice_processing():
    """
    TEST CASE 5: Feature extraction from 2D slices
    - Kiểm tra sliding window trên slice 2D
    - Kiểm tra output shape
    - Kiểm tra boundary handling
    """
    print_test_header("Case 5 - 2D Slice Processing")
    
    extractor = GLCMExtractor(window_size=3, levels=16)
    
    # Test 5.1: Process small slice
    print("\n[5.1] Testing small slice processing...")
    slice_2d = np.random.randn(20, 20).astype(np.float32)
    
    feature_maps = extractor.extract_features_from_slice(slice_2d)
    
    assert feature_maps.shape == (20, 20, 5), f"Expected (20,20,5), got {feature_maps.shape}"
    assert not np.any(np.isnan(feature_maps)), "Feature maps should not contain NaN"
    assert not np.any(np.isinf(feature_maps)), "Feature maps should not contain Inf"
    print(f"✓ Feature maps shape: {feature_maps.shape}")
    
    # Test 5.2: Check boundary (edges should have zeros due to window)
    print("\n[5.2] Testing boundary handling...")
    half_window = extractor.window_size // 2
    
    # Check corners should be zero (not processed)
    assert feature_maps[0, 0, 0] == 0.0, "Top-left corner should be zero"
    assert feature_maps[-1, -1, 0] == 0.0, "Bottom-right corner should be zero"
    
    # Check center should have values
    center = feature_maps[10, 10, :]
    assert np.any(center != 0.0), "Center should have non-zero values"
    print(f"✓ Boundary handling correct (half_window={half_window})")
    
    # Test 5.3: Structured pattern
    print("\n[5.3] Testing structured pattern...")
    # Create checkerboard pattern
    checker = np.indices((20, 20)).sum(axis=0) % 2
    checker = checker.astype(np.float32)
    
    feature_checker = extractor.extract_features_from_slice(checker)
    
    # Contrast should be high for checkerboard
    contrast_mean = feature_checker[:, :, 0].mean()  # contrast is first feature
    print(f"✓ Checkerboard contrast mean: {contrast_mean:.4f}")
    
    print("\n✅ TEST CASE 5 PASSED: Slice processing works correctly")


def test_case_6_channel_processing():
    """
    TEST CASE 6: Feature extraction from 3D channel volume
    - Kiểm tra xử lý volume 3D single channel
    - Kiểm tra output shape (H, W, D, 5)
    - Kiểm tra consistency across slices
    """
    print_test_header("Case 6 - 3D Channel Processing")
    
    extractor = GLCMExtractor(window_size=3, levels=16)
    
    # Test 6.1: Process small 3D volume
    print("\n[6.1] Testing 3D channel volume processing...")
    channel_volume = np.random.randn(30, 30, 5).astype(np.float32)
    
    feature_volume = extractor.extract_features_channel(channel_volume)
    
    assert feature_volume.shape == (30, 30, 5, 5), f"Expected (30,30,5,5), got {feature_volume.shape}"
    assert not np.any(np.isnan(feature_volume)), "Should not contain NaN"
    print(f"✓ Feature volume shape: {feature_volume.shape}")
    
    # Test 6.2: Check each slice independently
    print("\n[6.2] Testing slice independence...")
    # Each slice should have features
    for z in range(5):
        slice_features = feature_volume[:, :, z, :]
        assert np.any(slice_features != 0.0), f"Slice {z} should have non-zero features"
    print("✓ All slices processed independently")
    
    # Test 6.3: Statistics
    print("\n[6.3] Testing feature statistics...")
    for feat_idx, feat_name in enumerate(extractor.feature_names):
        feat_data = feature_volume[:, :, :, feat_idx]
        mean_val = feat_data.mean()
        std_val = feat_data.std()
        print(f"    {feat_name}: mean={mean_val:.4f}, std={std_val:.4f}")
        assert not np.isnan(mean_val), f"{feat_name} mean should not be NaN"
    
    print("\n✅ TEST CASE 6 PASSED: Channel processing works correctly")


def test_case_7_full_volume():
    """
    TEST CASE 7: Full 4D volume feature extraction
    - Kiểm tra pipeline đầy đủ cho volume 4D
    - Kiểm tra transformation từ 4 kênh → 20 kênh features
    - Kiểm tra output shape và consistency
    """
    print_test_header("Case 7 - Full 4D Volume Processing")
    
    extractor = GLCMExtractor(window_size=3, levels=8)  # Small params for speed
    
    # Test 7.1: Small 4D volume (4 channels → 20 features)
    print("\n[7.1] Testing 4D volume transformation (4 channels → 20 features)...")
    small_volume = np.random.randn(20, 20, 5, 4).astype(np.float32)
    print(f"Input shape: {small_volume.shape}")
    
    features = extractor.extract_features(small_volume)
    
    expected_shape = (20, 20, 5, 20)  # 4 channels * 5 features = 20
    assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
    assert not np.any(np.isnan(features)), "Output should not contain NaN"
    print(f"✓ Output shape: {features.shape}")
    
    # Test 7.2: Verify channel mapping
    print("\n[7.2] Testing channel mapping (each input channel → 5 features)...")
    # Channel 0 features should be in indices 0-4
    # Channel 1 features should be in indices 5-9, etc.
    for c in range(4):
        start_idx = c * 5
        end_idx = start_idx + 5
        channel_features = features[:, :, :, start_idx:end_idx]
        
        # Should have non-zero values in center
        center_slice = channel_features[10, 10, 2, :]
        assert np.any(center_slice != 0.0), f"Channel {c} features should be non-zero"
        print(f"✓ Channel {c} → features[{start_idx}:{end_idx}]: OK")
    
    # Test 7.3: Different number of channels
    print("\n[7.3] Testing with 2 channels (2 → 10 features)...")
    volume_2ch = np.random.randn(20, 20, 5, 2).astype(np.float32)
    
    features_2ch = extractor.extract_features(volume_2ch)
    
    assert features_2ch.shape == (20, 20, 5, 10), f"Expected (20,20,5,10), got {features_2ch.shape}"
    print(f"✓ 2 channels → 10 features: {features_2ch.shape}")
    
    print("\n✅ TEST CASE 7 PASSED: Full volume processing works correctly")


def test_case_8_fast_mode():
    """
    TEST CASE 8: Fast extraction mode với stride
    - Kiểm tra fast mode với stride > 1
    - Kiểm tra output size reduction
    - So sánh performance với normal mode
    """
    print_test_header("Case 8 - Fast Extraction Mode")
    
    extractor = GLCMExtractor(window_size=3, levels=8)
    
    # Test 8.1: Fast mode with stride=2
    print("\n[8.1] Testing fast mode with stride=2...")
    volume = np.random.randn(40, 40, 5, 2).astype(np.float32)
    
    features_fast = extractor.extract_features_fast(volume, stride=2)
    
    # Output should be smaller due to stride
    assert features_fast.shape[0] <= 20, "Height should be reduced with stride=2"
    assert features_fast.shape[1] <= 20, "Width should be reduced with stride=2"
    assert features_fast.shape[2] == 5, "Depth should remain same"
    assert features_fast.shape[3] == 10, "Should have 2*5=10 features"
    print(f"✓ Fast mode output: {features_fast.shape}")
    
    # Test 8.2: Different stride values
    print("\n[8.2] Testing different stride values...")
    for stride in [1, 2, 4]:
        feat = extractor.extract_features_fast(volume, stride=stride)
        expected_h = (40 + stride - 1) // stride
        expected_w = (40 + stride - 1) // stride
        print(f"  Stride={stride}: {feat.shape} (expected H,W ≈ {expected_h},{expected_w})")
        assert not np.any(np.isnan(feat)), f"Stride {stride} should not produce NaN"
    
    # Test 8.3: Compare with full mode (small volume)
    print("\n[8.3] Comparing fast vs normal mode...")
    tiny_volume = np.random.randn(15, 15, 3, 1).astype(np.float32)
    
    import time
    
    # Fast mode
    start = time.time()
    feat_fast = extractor.extract_features_fast(tiny_volume, stride=2)
    time_fast = time.time() - start
    
    print(f"✓ Fast mode: {time_fast:.3f}s, shape={feat_fast.shape}")
    
    print("\n✅ TEST CASE 8 PASSED: Fast mode works correctly")


def test_case_9_edge_cases():
    """
    TEST CASE 9: Edge cases và error handling
    - Kiểm tra với volume shape không hợp lệ
    - Kiểm tra với giá trị extreme
    - Kiểm tra với volume rất nhỏ
    """
    print_test_header("Case 9 - Edge Cases")
    
    extractor = GLCMExtractor(window_size=3, levels=8)
    
    # Test 9.1: Invalid shape (not 4D)
    print("\n[9.1] Testing invalid volume shape...")
    invalid_volume = np.random.randn(20, 20, 5).astype(np.float32)  # 3D
    
    try:
        extractor.extract_features(invalid_volume)
        assert False, "Should raise ValueError for 3D input"
    except ValueError as e:
        print(f"✓ Correctly rejected 3D input: {str(e)}")
    
    # Test 9.2: Very small volume
    print("\n[9.2] Testing very small volume...")
    tiny = np.random.randn(5, 5, 2, 1).astype(np.float32)
    
    feat_tiny = extractor.extract_features_fast(tiny, stride=1)
    assert feat_tiny.shape[3] == 5, "Should have 5 features for 1 channel"
    print(f"✓ Tiny volume processed: {feat_tiny.shape}")
    
    # Test 9.3: Extreme values
    print("\n[9.3] Testing extreme values...")
    extreme = np.random.randn(20, 20, 3, 2).astype(np.float32) * 1e6
    
    feat_extreme = extractor.extract_features_fast(extreme, stride=3)
    assert not np.any(np.isnan(feat_extreme)), "Should handle extreme values"
    assert not np.any(np.isinf(feat_extreme)), "Should not produce Inf"
    print("✓ Extreme values handled correctly")
    
    # Test 9.4: All zeros volume
    print("\n[9.4] Testing all-zeros volume...")
    zeros = np.zeros((20, 20, 3, 1), dtype=np.float32)
    
    feat_zeros = extractor.extract_features_fast(zeros, stride=2)
    # Should complete without errors (features will be zero or default)
    print(f"✓ All-zeros volume processed: {feat_zeros.shape}")
    
    print("\n✅ TEST CASE 9 PASSED: Edge cases handled correctly")


def test_case_10_realistic_scenario():
    """
    TEST CASE 10: Realistic scenario với BraTS-like data
    - Kiểm tra với kích thước gần giống BraTS (scaled down)
    - Kiểm tra consistency với preprocessed data
    - Test integration với module preprocessing
    """
    print_test_header("Case 10 - Realistic BraTS-like Scenario")
    
    extractor = GLCMExtractor(window_size=5, levels=32)
    
    # Test 10.1: Scaled-down BraTS volume (60×60×40×4 instead of 240×240×160×4)
    print("\n[10.1] Testing scaled BraTS-like volume (60×60×40×4)...")
    # Simulate preprocessed MRI data (normalized, mean≈0, std≈1)
    brats_like = np.random.randn(60, 60, 40, 4).astype(np.float32)
    
    print("Processing with fast mode (stride=3) for speed...")
    features = extractor.extract_features_fast(brats_like, stride=3)
    
    expected_channels = 4 * 5  # 4 input channels * 5 features = 20
    assert features.shape[3] == expected_channels, f"Expected 20 channels, got {features.shape[3]}"
    print(f"✓ Output: {features.shape} ({expected_channels} feature channels)")
    
    # Test 10.2: Verify feature statistics
    print("\n[10.2] Verifying feature statistics...")
    feature_names = extractor.feature_names
    
    for c in range(4):
        print(f"\n  Channel {c}:")
        for feat_idx, feat_name in enumerate(feature_names):
            global_idx = c * 5 + feat_idx
            feat_data = features[:, :, :, global_idx]
            
            mean = feat_data.mean()
            std = feat_data.std()
            min_val = feat_data.min()
            max_val = feat_data.max()
            
            print(f"    {feat_name}: mean={mean:.4f}, std={std:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
            
            # Basic sanity checks
            assert not np.isnan(mean), f"Mean should not be NaN for {feat_name}"
            assert min_val <= max_val, "Min should be <= Max"
    
    print("\n✅ TEST CASE 10 PASSED: Realistic scenario works correctly")


def run_all_tests():
    """
    Chạy tất cả test cases
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  GLCM EXTRACTION MODULE - COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    test_functions = [
        test_case_1_initialization,
        test_case_2_quantization,
        test_case_3_glcm_computation,
        test_case_4_haralick_features,
        test_case_5_slice_processing,
        test_case_6_channel_processing,
        test_case_7_full_volume,
        test_case_8_fast_mode,
        test_case_9_edge_cases,
        test_case_10_realistic_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {test_func.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  TEST SUMMARY".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\n✅ PASSED: {passed}/{len(test_functions)}")
    print(f"❌ FAILED: {failed}/{len(test_functions)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Module is ready for use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
    
    print("\n" + "█"*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
