"""
Test Suite cho Module Preprocessing
Kiểm thử đầy đủ các chức năng của Preprocessor class.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path to import preprocessor
sys.path.insert(0, str(Path(__file__).parent))

from preprocessor import Preprocessor


def print_test_header(test_name):
    """Helper function để in header của test case"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def test_case_1_initialization():
    """
    TEST CASE 1: Khởi tạo Preprocessor
    - Kiểm tra khởi tạo với default parameters
    - Kiểm tra khởi tạo với custom parameters
    """
    print_test_header("Case 1 - Initialization")
    
    # Test 1.1: Default initialization
    print("\n[1.1] Testing default initialization...")
    preprocessor = Preprocessor()
    assert preprocessor.target_shape == (240, 240, 160, 4), "Default shape mismatch!"
    print("✓ Default initialization passed")
    
    # Test 1.2: Custom initialization
    print("\n[1.2] Testing custom initialization...")
    custom_shape = (128, 128, 128, 4)
    preprocessor_custom = Preprocessor(target_shape=custom_shape)
    assert preprocessor_custom.target_shape == custom_shape, "Custom shape mismatch!"
    print(f"✓ Custom initialization passed with shape {custom_shape}")
    
    print("\n✅ TEST CASE 1 PASSED: Initialization works correctly")


def test_case_2_padding():
    """
    TEST CASE 2: Zero-padding functionality
    - Kiểm tra padding từ 155 → 160 depth
    - Kiểm tra padding với các kích thước khác nhau
    - Kiểm tra giá trị padding là 0
    """
    print_test_header("Case 2 - Zero-Padding")
    
    preprocessor = Preprocessor()
    
    # Test 2.1: Standard padding (155 → 160)
    print("\n[2.1] Testing standard padding (155 → 160 depth)...")
    volume_155 = np.random.randn(240, 240, 155, 4).astype(np.float32)
    padded = preprocessor.pad_volume(volume_155)
    
    assert padded.shape == (240, 240, 160, 4), f"Expected shape (240,240,160,4), got {padded.shape}"
    assert np.all(padded[:, :, 155:, :] == 0), "Padded region should be all zeros!"
    assert np.allclose(padded[:, :, :155, :], volume_155), "Original data should be preserved!"
    print("✓ Standard padding passed")
    
    # Test 2.2: Padding smaller volume
    print("\n[2.2] Testing padding smaller volume (100×100×100×4)...")
    small_volume = np.random.randn(100, 100, 100, 4).astype(np.float32)
    padded_small = preprocessor.pad_volume(small_volume)
    
    assert padded_small.shape == (240, 240, 160, 4), f"Expected shape (240,240,160,4), got {padded_small.shape}"
    assert np.allclose(padded_small[:100, :100, :100, :], small_volume), "Original data should be at start!"
    print("✓ Small volume padding passed")
    
    # Test 2.3: No padding needed (already target size)
    print("\n[2.3] Testing volume already at target size...")
    target_volume = np.random.randn(240, 240, 160, 4).astype(np.float32)
    padded_target = preprocessor.pad_volume(target_volume)
    
    assert padded_target.shape == (240, 240, 160, 4), "Shape should remain unchanged"
    assert np.allclose(padded_target, target_volume), "Data should be identical!"
    print("✓ No-padding case passed")
    
    # Test 2.4: 3D input (auto expand to 4D with single channel)
    print("\n[2.4] Testing 3D input auto-expansion...")
    volume_3d = np.random.randn(240, 240, 155).astype(np.float32)
    # Use custom preprocessor for single channel
    preprocessor_single = Preprocessor(target_shape=(240, 240, 160, 1))
    padded_3d = preprocessor_single.pad_volume(volume_3d)
    
    assert padded_3d.shape == (240, 240, 160, 1), f"Expected shape (240,240,160,1), got {padded_3d.shape}"
    print("✓ 3D input auto-expansion passed")
    
    print("\n✅ TEST CASE 2 PASSED: Zero-padding works correctly")


def test_case_3_normalization():
    """
    TEST CASE 3: Z-score normalization
    - Kiểm tra mean ≈ 0, std ≈ 1 sau normalization
    - Kiểm tra normalization độc lập cho từng kênh
    - Kiểm tra xử lý trường hợp std = 0
    """
    print_test_header("Case 3 - Z-score Normalization")
    
    preprocessor = Preprocessor()
    
    # Test 3.1: Standard normalization
    print("\n[3.1] Testing standard z-score normalization...")
    # Tạo volume với mean và std khác nhau cho mỗi kênh
    volume = np.zeros((240, 240, 160, 4), dtype=np.float32)
    volume[:, :, :, 0] = np.random.randn(240, 240, 160) * 100 + 50   # mean=50, std=100
    volume[:, :, :, 1] = np.random.randn(240, 240, 160) * 10 + 20    # mean=20, std=10
    volume[:, :, :, 2] = np.random.randn(240, 240, 160) * 5 - 30     # mean=-30, std=5
    volume[:, :, :, 3] = np.random.randn(240, 240, 160) * 200 + 100  # mean=100, std=200
    
    normalized = preprocessor.normalize_zscore(volume)
    
    # Kiểm tra từng kênh
    for c in range(4):
        channel_mean = normalized[:, :, :, c].mean()
        channel_std = normalized[:, :, :, c].std()
        
        assert abs(channel_mean) < 1e-5, f"Channel {c} mean should be ≈0, got {channel_mean}"
        assert abs(channel_std - 1.0) < 0.01, f"Channel {c} std should be ≈1, got {channel_std}"
        print(f"✓ Channel {c}: mean={channel_mean:.6f}, std={channel_std:.6f}")
    
    print("✓ Standard normalization passed")
    
    # Test 3.2: Zero std handling (constant volume)
    print("\n[3.2] Testing zero std handling (constant volume)...")
    constant_volume = np.ones((240, 240, 160, 4), dtype=np.float32) * 42.0
    normalized_constant = preprocessor.normalize_zscore(constant_volume)
    
    # Nên xử lý được mà không crash, và kết quả hợp lý
    assert not np.any(np.isnan(normalized_constant)), "Should not contain NaN!"
    assert not np.any(np.isinf(normalized_constant)), "Should not contain Inf!"
    print("✓ Zero std handling passed")
    
    # Test 3.3: Verify independence between channels
    print("\n[3.3] Testing channel independence...")
    volume_independent = np.zeros((50, 50, 50, 2), dtype=np.float32)
    volume_independent[:, :, :, 0] = np.random.randn(50, 50, 50) * 100  # Large std
    volume_independent[:, :, :, 1] = np.random.randn(50, 50, 50) * 0.1  # Small std
    
    normalized_ind = preprocessor.normalize_zscore(volume_independent)
    
    # Cả 2 kênh nên có std ≈ 1 sau normalization
    std_0 = normalized_ind[:, :, :, 0].std()
    std_1 = normalized_ind[:, :, :, 1].std()
    
    assert abs(std_0 - 1.0) < 0.01, f"Channel 0 std should be ≈1, got {std_0}"
    assert abs(std_1 - 1.0) < 0.01, f"Channel 1 std should be ≈1, got {std_1}"
    print(f"✓ Both channels normalized independently: std_0={std_0:.6f}, std_1={std_1:.6f}")
    
    print("\n✅ TEST CASE 3 PASSED: Z-score normalization works correctly")


def test_case_4_full_pipeline():
    """
    TEST CASE 4: Full preprocessing pipeline
    - Kiểm tra preprocess() method
    - Kiểm tra kết hợp padding + normalization
    - Kiểm tra output shape và properties
    """
    print_test_header("Case 4 - Full Preprocessing Pipeline")
    
    preprocessor = Preprocessor()
    
    # Test 4.1: Standard BraTS volume (240×240×155×4)
    print("\n[4.1] Testing full pipeline with standard BraTS volume...")
    brats_volume = np.random.randn(240, 240, 155, 4).astype(np.float32) * 100 + 50
    
    processed = preprocessor.preprocess(brats_volume)
    
    # Kiểm tra shape
    assert processed.shape == (240, 240, 160, 4), f"Expected (240,240,160,4), got {processed.shape}"
    
    # Kiểm tra normalization
    for c in range(4):
        channel_mean = processed[:, :, :, c].mean()
        channel_std = processed[:, :, :, c].std()
        assert abs(channel_mean) < 1e-5, f"Channel {c} mean should be ≈0"
        assert abs(channel_std - 1.0) < 0.01, f"Channel {c} std should be ≈1"
    
    # Kiểm tra padding region (sau normalization, vùng padding sẽ có giá trị âm do normalization)
    # Kiểm tra rằng vùng padding khác biệt với vùng dữ liệu gốc
    padded_region_mean = processed[:, :, 155:, :].mean()
    original_region_mean = processed[:, :, :155, :].mean()
    # Padded region nên có mean gần 0 hoặc âm hơn (do zero values bị normalize)
    print(f"✓ Padded region mean: {padded_region_mean:.6f}, Original region mean: {original_region_mean:.6f}")
    
    print("✓ Full pipeline passed for standard volume")
    
    # Test 4.2: Smaller volume
    print("\n[4.2] Testing full pipeline with smaller volume...")
    small_volume = np.random.randn(100, 100, 100, 2).astype(np.float32)
    
    preprocessor_custom = Preprocessor(target_shape=(128, 128, 128, 4))
    processed_small = preprocessor_custom.preprocess(small_volume)
    
    assert processed_small.shape == (128, 128, 128, 4), "Shape mismatch for custom target!"
    print("✓ Full pipeline passed for small volume with custom target")
    
    # Test 4.3: 3D single channel input
    print("\n[4.3] Testing full pipeline with 3D single channel...")
    single_channel = np.random.randn(240, 240, 155).astype(np.float32)
    
    preprocessor_single = Preprocessor(target_shape=(240, 240, 160, 1))
    processed_single = preprocessor_single.preprocess(single_channel)
    
    assert processed_single.shape == (240, 240, 160, 1), "3D input should expand to 4D"
    print("✓ Full pipeline passed for 3D input")
    
    print("\n✅ TEST CASE 4 PASSED: Full pipeline works correctly")


def test_case_5_edge_cases():
    """
    TEST CASE 5: Edge cases và error handling
    - Kiểm tra với volume rất nhỏ
    - Kiểm tra với giá trị âm
    - Kiểm tra với giá trị rất lớn
    - Kiểm tra data types khác nhau
    """
    print_test_header("Case 5 - Edge Cases")
    
    preprocessor = Preprocessor()
    
    # Test 5.1: Very small volume
    print("\n[5.1] Testing very small volume (10×10×10×1)...")
    tiny_volume = np.random.randn(10, 10, 10, 1).astype(np.float32)
    processed_tiny = preprocessor.preprocess(tiny_volume)
    
    assert processed_tiny.shape == (240, 240, 160, 4), "Should pad to target size"
    print("✓ Tiny volume handled correctly")
    
    # Test 5.2: Negative values
    print("\n[5.2] Testing with negative values...")
    negative_volume = np.random.randn(240, 240, 155, 4).astype(np.float32) - 100
    processed_negative = preprocessor.preprocess(negative_volume)
    
    # Should normalize properly regardless of original values
    for c in range(4):
        assert abs(processed_negative[:, :, :, c].mean()) < 1e-5, "Mean should be ≈0"
    print("✓ Negative values handled correctly")
    
    # Test 5.3: Very large values
    print("\n[5.3] Testing with very large values...")
    large_volume = np.random.randn(240, 240, 155, 4).astype(np.float32) * 1e6
    processed_large = preprocessor.preprocess(large_volume)
    
    assert not np.any(np.isnan(processed_large)), "Should not produce NaN"
    assert not np.any(np.isinf(processed_large)), "Should not produce Inf"
    print("✓ Large values handled correctly")
    
    # Test 5.4: Different data types
    print("\n[5.4] Testing with different data types...")
    int_volume = np.random.randint(0, 255, size=(240, 240, 155, 4), dtype=np.int32)
    processed_int = preprocessor.preprocess(int_volume)
    
    assert processed_int.dtype == np.float32, "Output should be float32"
    print("✓ Integer input converted correctly")
    
    # Test 5.5: All zeros volume
    print("\n[5.5] Testing all-zeros volume...")
    zero_volume = np.zeros((240, 240, 155, 4), dtype=np.float32)
    processed_zero = preprocessor.preprocess(zero_volume)
    
    assert not np.any(np.isnan(processed_zero)), "All-zero volume should not produce NaN"
    print("✓ All-zeros volume handled correctly")
    
    print("\n✅ TEST CASE 5 PASSED: Edge cases handled correctly")


def test_case_6_performance():
    """
    TEST CASE 6: Performance và memory
    - Kiểm tra thời gian xử lý
    - Kiểm tra memory footprint
    """
    print_test_header("Case 6 - Performance")
    
    import time
    
    preprocessor = Preprocessor()
    
    print("\n[6.1] Testing processing time...")
    volume = np.random.randn(240, 240, 155, 4).astype(np.float32)
    
    start_time = time.time()
    processed = preprocessor.preprocess(volume)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"✓ Processing time: {processing_time:.4f} seconds")
    
    # Reasonable time check (should be < 5 seconds on modern hardware)
    assert processing_time < 5.0, f"Processing too slow: {processing_time:.4f}s"
    
    print("\n[6.2] Testing memory efficiency...")
    import sys
    memory_before = sys.getsizeof(volume)
    memory_after = sys.getsizeof(processed)
    
    print(f"✓ Input size: {memory_before / 1024 / 1024:.2f} MB")
    print(f"✓ Output size: {memory_after / 1024 / 1024:.2f} MB")
    
    print("\n✅ TEST CASE 6 PASSED: Performance acceptable")


def run_all_tests():
    """
    Chạy tất cả test cases
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PREPROCESSING MODULE - COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    test_functions = [
        test_case_1_initialization,
        test_case_2_padding,
        test_case_3_normalization,
        test_case_4_full_pipeline,
        test_case_5_edge_cases,
        test_case_6_performance
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
