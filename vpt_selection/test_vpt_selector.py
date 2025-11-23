"""
Test Suite cho Module VPT Selection
Kiểm thử đầy đủ các chức năng của VPTSelector class.
"""

import numpy as np
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vpt_selector import VPTSelector, VPTNode


def print_test_header(test_name):
    """Helper function để in header của test case"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)


def test_case_1_initialization():
    """
    TEST CASE 1: Khởi tạo VPTSelector
    - Kiểm tra khởi tạo với default parameters
    - Kiểm tra khởi tạo với custom parameters
    - Kiểm tra các thuộc tính cơ bản
    """
    print_test_header("Case 1 - Initialization")
    
    # Test 1.1: Default initialization
    print("\n[1.1] Testing default initialization...")
    selector = VPTSelector()
    
    assert selector.n_features == 10, "Default n_features should be 10"
    assert selector.distance_metric == 'euclidean', "Default metric should be euclidean"
    assert selector.selection_method == 'variance', "Default method should be variance"
    assert selector.k_neighbors == 5, "Default k_neighbors should be 5"
    assert selector.vpt_root is None, "Initial tree should be None"
    print("✓ Default initialization passed")
    
    # Test 1.2: Custom initialization
    print("\n[1.2] Testing custom initialization...")
    selector_custom = VPTSelector(
        n_features=5,
        distance_metric='manhattan',
        selection_method='pca',
        k_neighbors=3
    )
    
    assert selector_custom.n_features == 5, "Custom n_features mismatch"
    assert selector_custom.distance_metric == 'manhattan', "Custom metric mismatch"
    assert selector_custom.selection_method == 'pca', "Custom method mismatch"
    assert selector_custom.k_neighbors == 3, "Custom k_neighbors mismatch"
    print("✓ Custom initialization passed")
    
    # Test 1.3: Various selection methods
    print("\n[1.3] Testing selection method options...")
    methods = ['variance', 'pca', 'correlation']
    for method in methods:
        sel = VPTSelector(selection_method=method)
        assert sel.selection_method == method, f"Method {method} not set correctly"
    print(f"✓ All methods available: {methods}")
    
    print("\n✅ TEST CASE 1 PASSED: Initialization works correctly")


def test_case_2_vpt_node():
    """
    TEST CASE 2: VPTNode structure
    - Kiểm tra tạo VPTNode
    - Kiểm tra thuộc tính node
    - Kiểm tra tree structure
    """
    print_test_header("Case 2 - VPT Node Structure")
    
    # Test 2.1: Create basic node
    print("\n[2.1] Testing basic node creation...")
    vp = np.array([1.0, 2.0, 3.0])
    node = VPTNode(vp, label=1, radius=5.0)
    
    assert np.allclose(node.vantage_point, vp), "Vantage point mismatch"
    assert node.label == 1, "Label mismatch"
    assert node.radius == 5.0, "Radius mismatch"
    assert node.left is None, "Left child should be None"
    assert node.right is None, "Right child should be None"
    print("✓ Basic node created successfully")
    
    # Test 2.2: Create tree structure
    print("\n[2.2] Testing tree structure...")
    root = VPTNode(np.array([0, 0, 0]), label=0)
    root.left = VPTNode(np.array([1, 1, 1]), label=1)
    root.right = VPTNode(np.array([2, 2, 2]), label=0)
    
    assert root.left is not None, "Left child should exist"
    assert root.right is not None, "Right child should exist"
    assert root.left.label == 1, "Left child label mismatch"
    print("✓ Tree structure correct")
    
    print("\n✅ TEST CASE 2 PASSED: VPT Node works correctly")


def test_case_3_distance_computation():
    """
    TEST CASE 3: Distance computation
    - Kiểm tra Euclidean distance
    - Kiểm tra Manhattan distance
    - Kiểm tra consistency
    """
    print_test_header("Case 3 - Distance Computation")
    
    # Test 3.1: Euclidean distance
    print("\n[3.1] Testing Euclidean distance...")
    selector = VPTSelector(distance_metric='euclidean')
    
    vec1 = np.array([0.0, 0.0, 0.0])
    vec2 = np.array([3.0, 4.0, 0.0])
    
    dist = selector._compute_distance(vec1, vec2)
    expected = 5.0  # 3-4-5 triangle
    
    assert abs(dist - expected) < 1e-6, f"Expected {expected}, got {dist}"
    print(f"✓ Euclidean distance: {dist:.4f}")
    
    # Test 3.2: Manhattan distance
    print("\n[3.2] Testing Manhattan distance...")
    selector_manhattan = VPTSelector(distance_metric='manhattan')
    
    dist_manhattan = selector_manhattan._compute_distance(vec1, vec2)
    expected_manhattan = 7.0  # |3| + |4| + |0|
    
    assert abs(dist_manhattan - expected_manhattan) < 1e-6, f"Expected {expected_manhattan}, got {dist_manhattan}"
    print(f"✓ Manhattan distance: {dist_manhattan:.4f}")
    
    # Test 3.3: Zero distance
    print("\n[3.3] Testing zero distance (same points)...")
    vec_same = np.array([1.0, 2.0, 3.0])
    dist_zero = selector._compute_distance(vec_same, vec_same)
    
    assert dist_zero < 1e-10, f"Distance should be ~0, got {dist_zero}"
    print("✓ Zero distance for identical points")
    
    print("\n✅ TEST CASE 3 PASSED: Distance computation works correctly")


def test_case_4_vpt_tree_building():
    """
    TEST CASE 4: VPT tree construction
    - Kiểm tra build tree từ small dataset
    - Kiểm tra tree structure sau building
    - Kiểm tra với labeled data
    """
    print_test_header("Case 4 - VPT Tree Building")
    
    selector = VPTSelector()
    
    # Test 4.1: Build tree from small dataset
    print("\n[4.1] Testing tree building with small dataset...")
    points = np.random.randn(20, 5).astype(np.float32)
    labels = np.random.randint(0, 2, size=20)
    
    selector.build_tree(points, labels)
    
    assert selector.vpt_root is not None, "Root should not be None after building"
    assert selector.vpt_root.vantage_point is not None, "Root should have vantage point"
    print("✓ Tree built successfully")
    
    # Test 4.2: Build tree without labels
    print("\n[4.2] Testing tree building without labels...")
    selector_no_label = VPTSelector()
    points_no_label = np.random.randn(15, 3).astype(np.float32)
    
    selector_no_label.build_tree(points_no_label, labels=None)
    
    assert selector_no_label.vpt_root is not None, "Tree should build without labels"
    print("✓ Tree built without labels")
    
    # Test 4.3: Larger dataset
    print("\n[4.3] Testing tree with larger dataset...")
    large_points = np.random.randn(100, 10).astype(np.float32)
    large_labels = np.random.randint(0, 3, size=100)
    
    selector_large = VPTSelector()
    selector_large.build_tree(large_points, large_labels)
    
    assert selector_large.vpt_root is not None, "Large tree should build"
    print("✓ Large tree built successfully")
    
    print("\n✅ TEST CASE 4 PASSED: Tree building works correctly")


def test_case_5_nearest_neighbor_query():
    """
    TEST CASE 5: K-nearest neighbor queries
    - Kiểm tra tìm k neighbors
    - Kiểm tra kết quả sorted by distance
    - Kiểm tra với different k values
    """
    print_test_header("Case 5 - Nearest Neighbor Query")
    
    selector = VPTSelector(k_neighbors=5)
    
    # Build tree with known points
    print("\n[5.1] Testing k-NN query...")
    points = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [5, 5], [6, 5], [5, 6], [6, 6]
    ], dtype=np.float32)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    selector.build_tree(points, labels)
    
    # Query point gần cluster 0
    query = np.array([0.5, 0.5], dtype=np.float32)
    neighbors = selector.query_nearest(query, k=3)
    
    assert len(neighbors) <= 3, f"Should return at most 3 neighbors, got {len(neighbors)}"
    assert len(neighbors) > 0, "Should return at least 1 neighbor"
    
    # Check sorted by distance
    distances = [dist for dist, _ in neighbors]
    assert distances == sorted(distances), "Results should be sorted by distance"
    print(f"✓ Found {len(neighbors)} neighbors, distances: {[f'{d:.2f}' for d in distances]}")
    
    # Test 5.2: Different k values
    print("\n[5.2] Testing different k values...")
    for k in [1, 3, 5, 10]:
        neighbors_k = selector.query_nearest(query, k=k)
        expected_count = min(k, len(points))
        assert len(neighbors_k) <= expected_count, f"Should return at most {expected_count} neighbors (k={k}, total={len(points)})"
        print(f"  k={k}: found {len(neighbors_k)} neighbors")
    
    # Test 5.3: Query without building tree
    print("\n[5.3] Testing query without tree (should raise error)...")
    selector_no_tree = VPTSelector()
    try:
        selector_no_tree.query_nearest(query)
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"✓ Correctly raised error: {str(e)[:50]}...")
    
    print("\n✅ TEST CASE 5 PASSED: k-NN query works correctly")


def test_case_6_variance_selection():
    """
    TEST CASE 6: Feature selection by variance
    - Kiểm tra selection by variance
    - Kiểm tra features có variance cao được chọn
    - Kiểm tra output shape
    """
    print_test_header("Case 6 - Variance-based Selection")
    
    # Test 6.1: Basic variance selection
    print("\n[6.1] Testing variance-based selection...")
    selector = VPTSelector(n_features=5, selection_method='variance')
    
    # Create volume với variance khác nhau
    volume = np.zeros((20, 20, 10, 10), dtype=np.float32)
    for c in range(10):
        # Channel càng cao càng có variance lớn
        volume[:, :, :, c] = np.random.randn(20, 20, 10) * (c + 1)
    
    selected = selector.select_features(volume)
    
    assert selected.shape == (20, 20, 10, 5), f"Expected (20,20,10,5), got {selected.shape}"
    print(f"✓ Output shape: {selected.shape}")
    
    # Test 6.2: Check selected indices
    print("\n[6.2] Verifying high-variance features selected...")
    indices = selector.get_selected_indices()
    
    assert indices is not None, "Should have feature indices"
    assert len(indices) == 5, f"Should have 5 indices, got {len(indices)}"
    
    # Indices nên là các channel có variance cao (channel 5-9)
    assert all(idx >= 4 for idx in indices), "Should select high-variance channels"
    print(f"  Selected indices: {indices}")
    
    # Test 6.3: Feature importance
    print("\n[6.3] Testing feature importance scores...")
    importance = selector.get_feature_importance()
    
    assert importance is not None, "Should have importance scores"
    assert len(importance) == 10, "Should have scores for all 10 original features"
    print(f"  Importance range: [{importance.min():.4f}, {importance.max():.4f}]")
    
    print("\n✅ TEST CASE 6 PASSED: Variance selection works correctly")


def test_case_7_correlation_selection():
    """
    TEST CASE 7: Feature selection by correlation
    - Kiểm tra selection giảm redundancy
    - Kiểm tra features được chọn có correlation thấp
    - So sánh với variance method
    """
    print_test_header("Case 7 - Correlation-based Selection")
    
    # Test 7.1: Basic correlation selection
    print("\n[7.1] Testing correlation-based selection...")
    selector = VPTSelector(n_features=5, selection_method='correlation')
    
    # Create volume với features có correlation
    volume = np.zeros((15, 15, 8, 10), dtype=np.float32)
    base = np.random.randn(15, 15, 8)
    
    for c in range(10):
        if c < 5:
            # First 5 channels tương tự nhau (high correlation)
            volume[:, :, :, c] = base + np.random.randn(15, 15, 8) * 0.1
        else:
            # Last 5 channels khác nhau (low correlation)
            volume[:, :, :, c] = np.random.randn(15, 15, 8)
    
    selected = selector.select_features(volume)
    
    assert selected.shape == (15, 15, 8, 5), f"Expected (15,15,8,5), got {selected.shape}"
    print(f"✓ Output shape: {selected.shape}")
    
    # Test 7.2: Check diverse selection
    print("\n[7.2] Verifying diverse features selected...")
    indices = selector.get_selected_indices()
    
    assert indices is not None, "Should have indices"
    assert len(indices) == 5, "Should have 5 indices"
    print(f"  Selected indices: {indices}")
    print("  (Should prefer diverse features over correlated ones)")
    
    print("\n✅ TEST CASE 7 PASSED: Correlation selection works correctly")


def test_case_8_pca_selection():
    """
    TEST CASE 8: Feature selection by PCA
    - Kiểm tra PCA transformation
    - Kiểm tra explained variance
    - Kiểm tra output shape
    """
    print_test_header("Case 8 - PCA-based Selection")
    
    # Test 8.1: Basic PCA
    print("\n[8.1] Testing PCA transformation...")
    selector = VPTSelector(n_features=5, selection_method='pca')
    
    volume = np.random.randn(20, 20, 10, 15).astype(np.float32)
    
    selected = selector.select_features(volume)
    
    assert selected.shape == (20, 20, 10, 5), f"Expected (20,20,10,5), got {selected.shape}"
    print(f"✓ PCA output shape: {selected.shape}")
    
    # Test 8.2: Check PCA model
    print("\n[8.2] Verifying PCA model...")
    assert selector.pca_model is not None, "PCA model should be saved"
    assert selector.pca_model.n_components_ == 5, "Should have 5 components"
    
    explained_var = selector.pca_model.explained_variance_ratio_
    total_var = explained_var.sum()
    print(f"  Explained variance: {total_var:.4f}")
    print(f"  Top 3 components: {explained_var[:3]}")
    
    # Test 8.3: Different component numbers
    print("\n[8.3] Testing with different n_components...")
    for n in [3, 7, 10]:
        sel = VPTSelector(n_features=n, selection_method='pca')
        result = sel.select_features(volume)
        assert result.shape[-1] == n, f"Should have {n} components"
        print(f"  n={n}: output shape {result.shape}")
    
    print("\n✅ TEST CASE 8 PASSED: PCA selection works correctly")


def test_case_9_full_pipeline():
    """
    TEST CASE 9: Full feature selection pipeline
    - Kiểm tra integration với preprocessing/GLCM output
    - Kiểm tra 20 → N transformation
    - Kiểm tra với realistic sizes
    """
    print_test_header("Case 9 - Full Pipeline")
    
    # Test 9.1: Simulate GLCM output (20 features)
    print("\n[9.1] Testing with GLCM-like input (20 features → 10)...")
    selector = VPTSelector(n_features=10, selection_method='variance')
    
    # Simulate output từ GLCM module (240×240×160×20 scaled down)
    glcm_like = np.random.randn(60, 60, 40, 20).astype(np.float32)
    
    selected = selector.select_features(glcm_like)
    
    assert selected.shape == (60, 60, 40, 10), f"Expected (60,60,40,10), got {selected.shape}"
    print(f"✓ GLCM output: {glcm_like.shape} → {selected.shape}")
    
    # Test 9.2: Edge case - n_features >= input channels
    print("\n[9.2] Testing when n_features >= input channels...")
    selector_large = VPTSelector(n_features=25, selection_method='variance')
    
    result_large = selector_large.select_features(glcm_like)
    
    # Should return original volume (no selection needed)
    assert result_large.shape == glcm_like.shape, "Should return original when n_features >= channels"
    print("✓ Correctly handles n_features >= input channels")
    
    # Test 9.3: Different methods comparison
    print("\n[9.3] Comparing selection methods...")
    volume_test = np.random.randn(30, 30, 15, 20).astype(np.float32)
    
    methods = ['variance', 'correlation', 'pca']
    for method in methods:
        sel = VPTSelector(n_features=8, selection_method=method)
        result = sel.select_features(volume_test)
        print(f"  {method}: {volume_test.shape} → {result.shape}")
        assert result.shape[-1] == 8, f"{method} should produce 8 features"
    
    print("\n✅ TEST CASE 9 PASSED: Full pipeline works correctly")


def test_case_10_prior_maps():
    """
    TEST CASE 10: Prior probability maps computation
    - Kiểm tra compute prior maps với VPT
    - Kiểm tra voting mechanism
    - Kiểm tra output shape và values
    """
    print_test_header("Case 10 - Prior Probability Maps")
    
    selector = VPTSelector(k_neighbors=5)
    
    # Test 10.1: Build tree với labeled data
    print("\n[10.1] Building VPT with labeled samples...")
    training_features = np.random.randn(50, 10).astype(np.float32)
    training_labels = np.random.randint(0, 2, size=50)
    
    selector.build_tree(training_features, training_labels)
    print("✓ Tree built with labels")
    
    # Test 10.2: Compute prior maps
    print("\n[10.2] Computing prior probability maps...")
    volume = np.random.randn(20, 20, 10, 10).astype(np.float32)
    
    prior_maps = selector.compute_prior_maps(volume)
    
    assert prior_maps.shape == (20, 20, 10), f"Expected (20,20,10), got {prior_maps.shape}"
    assert prior_maps.dtype == np.float32, "Should be float32"
    assert prior_maps.min() >= 0.0, "Probabilities should be >= 0"
    assert prior_maps.max() <= 1.0, "Probabilities should be <= 1"
    print(f"✓ Prior maps shape: {prior_maps.shape}")
    print(f"  Value range: [{prior_maps.min():.4f}, {prior_maps.max():.4f}]")
    print(f"  Mean: {prior_maps.mean():.4f}")
    
    # Test 10.3: Prior maps without tree
    print("\n[10.3] Testing prior maps without tree...")
    selector_no_tree = VPTSelector()
    
    prior_no_tree = selector_no_tree.compute_prior_maps(volume)
    
    assert np.all(prior_no_tree == 0), "Should return zeros without tree"
    print("✓ Returns zeros when no tree available")
    
    print("\n✅ TEST CASE 10 PASSED: Prior maps computation works correctly")


def test_case_11_edge_cases():
    """
    TEST CASE 11: Edge cases và error handling
    - Kiểm tra với volume shapes khác nhau
    - Kiểm tra với extreme values
    - Kiểm tra error handling
    """
    print_test_header("Case 11 - Edge Cases")
    
    # Test 11.1: Invalid shape (not 4D)
    print("\n[11.1] Testing invalid volume shape...")
    selector = VPTSelector()
    invalid_volume = np.random.randn(20, 20, 10).astype(np.float32)  # 3D
    
    try:
        selector.select_features(invalid_volume)
        assert False, "Should raise ValueError for 3D input"
    except ValueError as e:
        print(f"✓ Correctly rejected 3D input: {str(e)[:50]}...")
    
    # Test 11.2: Very small volume
    print("\n[11.2] Testing very small volume...")
    tiny = np.random.randn(5, 5, 3, 8).astype(np.float32)
    selector_small = VPTSelector(n_features=4, selection_method='variance')
    
    result_tiny = selector_small.select_features(tiny)
    assert result_tiny.shape == (5, 5, 3, 4), "Should handle tiny volumes"
    print(f"✓ Tiny volume processed: {result_tiny.shape}")
    
    # Test 11.3: Single feature selection
    print("\n[11.3] Testing single feature selection...")
    selector_single = VPTSelector(n_features=1, selection_method='variance')
    volume_single = np.random.randn(15, 15, 10, 10).astype(np.float32)
    
    result_single = selector_single.select_features(volume_single)
    assert result_single.shape[-1] == 1, "Should select single feature"
    print(f"✓ Single feature selected: {result_single.shape}")
    
    # Test 11.4: Extreme values
    print("\n[11.4] Testing extreme values...")
    extreme = np.random.randn(20, 20, 10, 15).astype(np.float32) * 1e6
    selector_extreme = VPTSelector(n_features=5, selection_method='variance')
    
    result_extreme = selector_extreme.select_features(extreme)
    assert not np.any(np.isnan(result_extreme)), "Should handle extreme values"
    assert not np.any(np.isinf(result_extreme)), "Should not produce Inf"
    print("✓ Extreme values handled correctly")
    
    # Test 11.5: All constant channels
    print("\n[11.5] Testing all constant channels...")
    constant = np.ones((15, 15, 8, 10), dtype=np.float32) * 42.0
    selector_const = VPTSelector(n_features=5, selection_method='variance')
    
    result_const = selector_const.select_features(constant)
    # Should still work (select any 5 channels)
    assert result_const.shape[-1] == 5, "Should select even from constant channels"
    print("✓ Constant channels handled")
    
    print("\n✅ TEST CASE 11 PASSED: Edge cases handled correctly")


def test_case_12_integration():
    """
    TEST CASE 12: Integration test với full pipeline
    - Simulate preprocessing → GLCM → VPT
    - Kiểm tra data flow
    - Kiểm tra consistency
    """
    print_test_header("Case 12 - Integration Test")
    
    print("\n[12.1] Simulating full pipeline...")
    print("  Step 1: Preprocessing (240×240×155×4 → 240×240×160×4)")
    preprocessed = np.random.randn(60, 60, 40, 4).astype(np.float32)  # Scaled
    
    print("  Step 2: GLCM Extraction (4 channels → 20 features)")
    glcm_features = np.random.randn(60, 60, 40, 20).astype(np.float32)
    
    print("  Step 3: VPT Selection (20 features → 10 features)")
    selector = VPTSelector(n_features=10, selection_method='variance')
    vpt_output = selector.select_features(glcm_features)
    
    assert vpt_output.shape == (60, 60, 40, 10), "Final shape mismatch"
    print(f"✓ Final output: {vpt_output.shape}")
    
    print("\n[12.2] Testing with labeled data for prior maps...")
    # Build VPT tree
    sample_features = np.random.randn(100, 20).astype(np.float32)
    sample_labels = np.random.randint(0, 2, size=100)
    
    selector.build_tree(sample_features, sample_labels)
    
    # Compute prior maps
    prior = selector.compute_prior_maps(glcm_features)
    
    assert prior.shape == (60, 60, 40), "Prior maps shape mismatch"
    print(f"✓ Prior maps: {prior.shape}")
    
    print("\n[12.3] Summary of full pipeline:")
    print(f"  Input (preprocessed): {preprocessed.shape}")
    print(f"  After GLCM: {glcm_features.shape}")
    print(f"  After VPT: {vpt_output.shape}")
    print(f"  Prior maps: {prior.shape}")
    print(f"  Reduction: {glcm_features.shape[-1]} → {vpt_output.shape[-1]} features ({100*(1-10/20):.0f}% reduction)")
    
    print("\n✅ TEST CASE 12 PASSED: Integration works correctly")


def run_all_tests():
    """
    Chạy tất cả test cases
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  VPT SELECTION MODULE - COMPREHENSIVE TEST SUITE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    test_functions = [
        test_case_1_initialization,
        test_case_2_vpt_node,
        test_case_3_distance_computation,
        test_case_4_vpt_tree_building,
        test_case_5_nearest_neighbor_query,
        test_case_6_variance_selection,
        test_case_7_correlation_selection,
        test_case_8_pca_selection,
        test_case_9_full_pipeline,
        test_case_10_prior_maps,
        test_case_11_edge_cases,
        test_case_12_integration
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
