"""
Simple test to check if postprocessing hangs
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from postprocessing import remove_small_components

print("Testing remove_small_components with small data...")
mask = torch.zeros(10, 10, 10, dtype=torch.long)
mask[2:5, 2:5, 2:5] = 1  # Large component
mask[7:8, 7:8, 7:8] = 1  # Small component (1 voxel)

print("Input created")
result = remove_small_components(mask, min_size=10)
print(f"Result: {result.shape}")
print("✓ Test passed!")
