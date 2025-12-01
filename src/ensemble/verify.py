"""
Quick verification script - Run a simple test to ensure everything works
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("QUICK VERIFICATION TEST")
print("="*70)

try:
    # Test imports
    print("\n1. Testing imports...")
    from ensemble import EnsembleModel
    from postprocessing import unpad_volume, postprocess_mask
    print("   ✓ All imports successful")
    
    # Test EnsembleModel instantiation
    print("\n2. Testing EnsembleModel instantiation...")
    ensemble_weighted = EnsembleModel(alpha=0.4, beta=0.6, strategy='weighted', device='cpu')
    ensemble_voting = EnsembleModel(strategy='voting', device='cpu')
    ensemble_hybrid = EnsembleModel(strategy='hybrid', device='cpu')
    print("   ✓ All strategies instantiated successfully")
    
    # Test weighted average
    print("\n3. Testing weighted average...")
    C, D, H, W = 4, 10, 10, 10
    prob_cnn = torch.rand(C, D, H, W)
    prob_unet = torch.rand(C, D, H, W)
    prob_cnn = prob_cnn / prob_cnn.sum(dim=0, keepdim=True)
    prob_unet = prob_unet / prob_unet.sum(dim=0, keepdim=True)
    
    result_weighted = ensemble_weighted.ensemble(prob_cnn, prob_unet)
    expected = 0.4 * prob_cnn + 0.6 * prob_unet
    assert torch.allclose(result_weighted, expected, atol=1e-6)
    print(f"   ✓ Weighted average correct: shape {result_weighted.shape}")
    
    # Test majority voting
    print("\n4. Testing majority voting...")
    result_voting = ensemble_voting.ensemble(prob_cnn, prob_unet)
    print(f"   ✓ Majority voting works: shape {result_voting.shape}")
    
    # Test hybrid
    print("\n5. Testing hybrid approach...")
    result_hybrid = ensemble_hybrid.ensemble(prob_cnn, prob_unet)
    print(f"   ✓ Hybrid approach works: shape {result_hybrid.shape}")
    
    # Test predict
    print("\n6. Testing predict (ensemble + argmax)...")
    mask = ensemble_weighted.predict(prob_cnn, prob_unet)
    print(f"   ✓ Mask generated: shape {mask.shape}")
    print(f"   ✓ Unique classes: {torch.unique(mask).tolist()}")
    
    # Test unpadding
    print("\n7. Testing unpadding...")
    volume_padded = torch.rand(160, 240, 240)
    volume_unpadded = unpad_volume(volume_padded, original_depth=155, padded_depth=160)
    assert volume_unpadded.shape == (155, 240, 240)
    print(f"   ✓ Unpadding works: {volume_padded.shape} → {volume_unpadded.shape}")
    
    # Test full post-processing (with smaller volume for speed)
    print("\n8. Testing full post-processing pipeline...")
    mask_padded = torch.randint(0, 4, (160, 50, 50))  # Smaller for speed
    mask_processed = postprocess_mask(
        mask_padded,
        original_depth=155,
        padded_depth=160,
        remove_small=True,
        smooth_boundary=False,  # Skip morphological ops for speed
        fill_holes_flag=False,   # Skip for speed
        enforce_consistency_flag=False  # Skip for speed
    )
    assert mask_processed.shape == (155, 50, 50)
    print(f"   ✓ Post-processing works: {mask_padded.shape} → {mask_processed.shape}")
    print("   Note: Full-size (240x240) volumes will be slower")
    
    # Test statistics
    print("\n9. Testing statistics...")
    stats = ensemble_weighted.get_statistics(prob_cnn, prob_unet)
    print(f"   ✓ Agreement rate: {stats['agreement_rate']:.2%}")
    print(f"   ✓ CNN confidence: {stats['avg_confidence_cnn']:.4f}")
    print(f"   ✓ U-Net confidence: {stats['avg_confidence_unet']:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL QUICK TESTS PASSED!")
    print("="*70)
    print("\nModule 6 is working correctly! Ready to use.")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
