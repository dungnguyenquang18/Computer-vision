"""Quick test to verify basic functionality"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from unet3d_model import UNet3DModel, UNet3DNet

print("=" * 80)
print("QUICK TEST SUITE FOR 3D U-NET")
print("=" * 80)

# Test 1: Model creation
print("\n[TEST 1] Creating model...")
try:
    model = UNet3DModel(input_shape=(64, 64, 64, 4), num_classes=4, base_filters=16)
    print("✓ Model created successfully")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n[TEST 2] Testing forward pass...")
try:
    x = torch.randn(1, 4, 64, 64, 64)
    output = model.model(x)
    expected_shape = (1, 4, 64, 64, 64)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"✓ Forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 3: predict_patch
print("\n[TEST 3] Testing predict_patch...")
try:
    patch = np.random.randn(64, 64, 64, 4).astype(np.float32)
    prediction = model.predict_patch(patch)
    expected_shape = (64, 64, 64, 4)
    assert prediction.shape == expected_shape, f"Expected {expected_shape}, got {prediction.shape}"
    assert np.all(prediction >= 0) and np.all(prediction <= 1), "Probabilities should be in [0,1]"
    prob_sums = np.sum(prediction, axis=-1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities should sum to 1"
    print(f"✓ predict_patch works correctly: {prediction.shape}")
except Exception as e:
    print(f"✗ predict_patch failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Training
print("\n[TEST 4] Testing training with 1 epoch...")
try:
    X_train = np.random.randn(2, 64, 64, 64, 4).astype(np.float32)
    y_train = np.random.randint(0, 4, size=(2, 64, 64, 64)).astype(np.int64)
    
    history = model.train((X_train, y_train), epochs=1, batch_size=1)
    
    assert 'loss' in history, "History should contain 'loss'"
    assert 'dice' in history, "History should contain 'dice'"
    assert len(history['loss']) == 1, "Should have 1 epoch of history"
    print(f"✓ Training works: loss={history['loss'][0]:.4f}, dice={history['dice'][0]:.4f}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: get_segmentation_mask
print("\n[TEST 5] Testing get_segmentation_mask...")
try:
    prob_map = np.random.rand(64, 64, 64, 4).astype(np.float32)
    prob_map = prob_map / np.sum(prob_map, axis=-1, keepdims=True)
    
    mask = model.get_segmentation_mask(prob_map)
    
    assert mask.shape == (64, 64, 64), f"Expected (64,64,64), got {mask.shape}"
    unique_values = set(np.unique(mask))
    valid_labels = {0, 1, 2, 4}
    assert unique_values.issubset(valid_labels), f"Invalid labels: {unique_values}"
    print(f"✓ get_segmentation_mask works: unique labels = {unique_values}")
except Exception as e:
    print(f"✗ get_segmentation_mask failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Save and load
print("\n[TEST 6] Testing save/load...")
try:
    save_path = "test_quick_model.pth"
    model.save_model(save_path)
    
    model2 = UNet3DModel(input_shape=(64, 64, 64, 4), num_classes=4, base_filters=16)
    model2.load_model(save_path)
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
    
    print(f"✓ Save/load works correctly")
except Exception as e:
    print(f"✗ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL QUICK TESTS PASSED!")
print("=" * 80)
