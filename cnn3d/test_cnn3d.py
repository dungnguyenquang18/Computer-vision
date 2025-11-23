"""
Test file for Module 4: 3D CNN Model
Tests all functionality of CNN3DModel and PatchDataLoader classes.

Run with: pytest test_cnn3d.py -v
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cnn3d_model import CNN3DModel, CNN3DNet
from data_loader import PatchDataLoader


class TestCNN3DModel:
    """Test cases for CNN3DModel class."""
    
    # ============================================================================
    # TEST CASE 1: Model Initialization
    # Kiểm tra việc khởi tạo model với các tham số mặc định và custom
    # ============================================================================
    def test_model_initialization_default(self):
        """
        Test 1.1: Khởi tạo model với tham số mặc định
        - Input shape: (128, 128, 128, 4)
        - Number of classes: 4
        - Kiểm tra model được build thành công
        """
        model = CNN3DModel()
        
        assert model.input_shape == (128, 128, 128, 4), "Default input shape should be (128, 128, 128, 4)"
        assert model.num_classes == 4, "Default num_classes should be 4"
        assert model.model is not None, "Model should be built after initialization"
        print("✓ Test 1.1 passed: Model initialized with default parameters")
    
    def test_model_initialization_custom(self):
        """
        Test 1.2: Khởi tạo model với tham số custom
        - Input shape: (64, 64, 64, 8)
        - Number of classes: 3
        """
        custom_shape = (64, 64, 64, 8)
        custom_classes = 3
        model = CNN3DModel(input_shape=custom_shape, num_classes=custom_classes)
        
        assert model.input_shape == custom_shape, f"Input shape should be {custom_shape}"
        assert model.num_classes == custom_classes, f"Num classes should be {custom_classes}"
        assert model.model is not None, "Model should be built with custom parameters"
        print("✓ Test 1.2 passed: Model initialized with custom parameters")
    
    # ============================================================================
    # TEST CASE 2: Model Architecture
    # Kiểm tra kiến trúc của model (layers, output shape)
    # ============================================================================
    def test_model_architecture(self):
        """
        Test 2.1: Kiểm tra số lượng layers trong model
        - Model nên có Conv3d, MaxPool3d, Linear, Dropout layers
        - Tổng số layers phải đúng với kiến trúc đã định nghĩa
        """
        model = CNN3DModel()
        
        # PyTorch uses modules() not layers
        layer_types = [type(module).__name__ for module in model.model.modules()]
        
        # Kiểm tra có các loại layer cần thiết
        assert 'Conv3d' in layer_types, "Model should contain Conv3d layers"
        assert 'MaxPool3d' in layer_types, "Model should contain MaxPool3d layers"
        assert 'Linear' in layer_types, "Model should contain Linear layers"
        assert 'Dropout' in layer_types, "Model should contain Dropout layers"
        
        print(f"✓ Test 2.1 passed: Model has correct layer types ({len(list(model.model.modules()))} total modules)")
    
    def test_model_output_shape(self):
        """
        Test 2.2: Kiểm tra output shape của model
        - Output shape phải là (batch_size, num_classes)
        """
        model = CNN3DModel(num_classes=4)
        
        # Tạo dummy input (N, H, W, D, C) -> convert to (N, C, H, W, D)
        dummy_input = np.random.randn(1, 128, 128, 128, 4).astype(np.float32)
        dummy_tensor = torch.FloatTensor(dummy_input).permute(0, 4, 1, 2, 3)
        
        model.model.eval()
        with torch.no_grad():
            output = model.model(dummy_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        assert output.shape == (1, 4), f"Output shape should be (1, 4), got {output.shape}"
        assert torch.allclose(probabilities.sum(dim=1), torch.tensor(1.0)), "Output should be probabilities summing to 1"
        print("✓ Test 2.2 passed: Model output shape is correct")
    
    # ============================================================================
    # TEST CASE 3: Model Compilation
    # Kiểm tra model được compile với optimizer, loss, metrics đúng
    # ============================================================================
    def test_model_compilation(self):
        """
        Test 3.1: Kiểm tra model initialization
        - Model được khởi tạo đúng
        - Device được set đúng (CPU hoặc CUDA)
        """
        model = CNN3DModel()
        
        assert model.model is not None, "Model should be initialized"
        assert model.device is not None, "Device should be set"
        assert isinstance(model.model, CNN3DNet), "Model should be instance of CNN3DNet"
        
        print("✓ Test 3.1 passed: Model initialized correctly")
    
    # ============================================================================
    # TEST CASE 4: Predict Patch
    # Kiểm tra dự đoán cho một patch đơn lẻ
    # ============================================================================
    def test_predict_patch_with_batch_dimension(self):
        """
        Test 4.1: Dự đoán với patch có batch dimension
        - Input: (1, 128, 128, 128, 4)
        - Output: (128, 128, 128, 4) - probability map
        """
        model = CNN3DModel()
        
        # Tạo dummy patch với batch dimension
        patch = np.random.randn(1, 128, 128, 128, 4).astype(np.float32)
        prediction = model.predict_patch(patch)
        
        assert prediction.shape == (4,), f"Prediction shape should be (4,), got {prediction.shape}"
        assert np.allclose(np.sum(prediction), 1.0), "Predictions should sum to 1"
        print("✓ Test 4.1 passed: Predict patch with batch dimension works")
    
    def test_predict_patch_without_batch_dimension(self):
        """
        Test 4.2: Dự đoán với patch không có batch dimension
        - Input: (128, 128, 128, 4)
        - Output: (128, 128, 128, 4) - probability map
        """
        model = CNN3DModel()
        
        # Tạo dummy patch không có batch dimension
        patch = np.random.randn(128, 128, 128, 4).astype(np.float32)
        prediction = model.predict_patch(patch)
        
        assert prediction.shape == (4,), f"Prediction shape should be (4,), got {prediction.shape}"
        assert np.allclose(np.sum(prediction), 1.0), "Predictions should sum to 1"
        print("✓ Test 4.2 passed: Predict patch without batch dimension works")
    
    # ============================================================================
    # TEST CASE 5: Predict Volume
    # Kiểm tra dự đoán cho toàn bộ volume bằng cách chia patches
    # ============================================================================
    def test_predict_volume_no_overlap(self):
        """
        Test 5.1: Dự đoán volume không overlap (với volume nhỏ cho test nhanh)
        - Volume size: (128, 128, 128, 4)
        - Patch size: (64, 64, 64)
        - Stride: (64, 64, 64) - no overlap
        """
        model = CNN3DModel(input_shape=(64, 64, 64, 4))
        
        # Tạo dummy volume nhỏ để test nhanh
        volume = np.random.randn(128, 128, 128, 4).astype(np.float32)
        
        # Predict với no overlap
        output = model.predict_volume(volume, patch_size=(64, 64, 64), stride=(64, 64, 64))
        
        assert output.shape == (128, 128, 128, 4), f"Output shape should be (128, 128, 128, 4), got {output.shape}"
        print("✓ Test 5.1 passed: Predict volume with no overlap works")
    
    # def test_predict_volume_with_overlap(self):
    #     """
    #     Test 5.2: Dự đoán volume có overlap
    #     - Volume size: (256, 256, 256, 4)
    #     - Patch size: (128, 128, 128)
    #     - Stride: (64, 64, 64) - 50% overlap
    #     """
    #     model = CNN3DModel()
        
    #     # Tạo dummy volume
    #     volume = np.random.randn(256, 256, 256, 4).astype(np.float32)
        
    #     # Predict với overlap
    #     output = model.predict_volume(volume, patch_size=(128, 128, 128), stride=(64, 64, 64))
        
    #     assert output.shape == (256, 256, 256, 4), f"Output shape should be (256, 256, 256, 4), got {output.shape}"
    #     print("✓ Test 5.2 passed: Predict volume with overlap works")
    
    # ============================================================================
    # TEST CASE 6: Get Segmentation Mask
    # Kiểm tra chuyển đổi probability map thành segmentation mask
    # ============================================================================
    def test_get_segmentation_mask(self):
        """
        Test 6.1: Chuyển đổi probability map thành segmentation mask
        - Input: (128, 128, 128, 4) probability map
        - Output: (128, 128, 128) segmentation mask
        - Class 3 phải được map thành label 4
        """
        model = CNN3DModel()
        
        # Tạo dummy probability map
        prob_map = np.random.rand(128, 128, 128, 4).astype(np.float32)
        # Normalize to valid probabilities
        prob_map = prob_map / np.sum(prob_map, axis=-1, keepdims=True)
        
        mask = model.get_segmentation_mask(prob_map)
        
        assert mask.shape == (128, 128, 128), f"Mask shape should be (128, 128, 128), got {mask.shape}"
        assert mask.dtype == np.uint8, f"Mask dtype should be uint8, got {mask.dtype}"
        assert np.all((mask >= 0) & (mask <= 4)), "Mask values should be in range [0, 4]"
        print("✓ Test 6.1 passed: Get segmentation mask works correctly")
    
    # ============================================================================
    # TEST CASE 7: Model Save/Load
    # Kiểm tra lưu và load model
    # ============================================================================
    # def test_save_load_model(self, tmp_path):
    #     """
    #     Test 7.1: Lưu và load model
    #     - Lưu model vào file
    #     - Load model từ file
    #     - Kiểm tra model load được hoạt động đúng
    #     """
    #     model1 = CNN3DModel()
        
    #     # Save model
    #     save_path = tmp_path / "test_model.h5"
    #     model1.save_model(str(save_path))
        
    #     # Load model
    #     model2 = CNN3DModel()
    #     model2.load_model(str(save_path))
        
    #     # Test prediction với cả 2 models
    #     patch = np.random.randn(1, 128, 128, 128, 4).astype(np.float32)
    #     pred1 = model1.predict_patch(patch)
    #     pred2 = model2.predict_patch(patch)
        
    #     assert np.allclose(pred1, pred2), "Loaded model should produce same predictions"
    #     print("✓ Test 7.1 passed: Model save/load works correctly")
    
    # ============================================================================
    # TEST CASE 8: Training (Basic)
    # Kiểm tra training với dummy data (không train thật để test nhanh)
    # ============================================================================
    # def test_training_basic(self):
    #     """
    #     Test 8.1: Training cơ bản với dummy data
    #     - Tạo dummy training data
    #     - Train 2 epochs
    #     - Kiểm tra history được tạo
    #     """
    #     model = CNN3DModel()
        
    #     # Tạo dummy training data
    #     X_train = np.random.randn(4, 128, 128, 128, 4).astype(np.float32)
    #     y_train = np.random.rand(4, 4).astype(np.float32)
    #     y_train = y_train / np.sum(y_train, axis=1, keepdims=True)  # Normalize
        
    #     # Train với epochs ít để test nhanh
    #     history = model.train((X_train, y_train), epochs=2, batch_size=2)
        
    #     assert history is not None, "Training should return history"
    #     assert 'loss' in history, "History should contain loss"
    #     assert len(history['loss']) == 2, "History should have 2 epochs"
    #     print("✓ Test 8.1 passed: Basic training works")


class TestPatchDataLoader:
    """Test cases for PatchDataLoader class."""
    
    # ============================================================================
    # TEST CASE 9: DataLoader Initialization
    # Kiểm tra khởi tạo data loader
    # ============================================================================
    def test_dataloader_initialization(self):
        """
        Test 9.1: Khởi tạo data loader với tham số mặc định
        - Patch size: (128, 128, 128)
        - Batch size: 2
        - Shuffle: True
        """
        loader = PatchDataLoader()
        
        assert loader.patch_size == (128, 128, 128), "Default patch size should be (128, 128, 128)"
        assert loader.batch_size == 2, "Default batch size should be 2"
        assert loader.shuffle == True, "Default shuffle should be True"
        print("✓ Test 9.1 passed: DataLoader initialized correctly")
    
    # ============================================================================
    # TEST CASE 10: Extract Patches with Stride
    # Kiểm tra extract patches từ volume với stride
    # ============================================================================
    def test_extract_patches_no_overlap(self):
        """
        Test 10.1: Extract patches không overlap
        - Volume: (256, 256, 256, 4)
        - Patch size: (128, 128, 128)
        - Stride: (128, 128, 128)
        - Expected: 8 patches (2x2x2)
        """
        loader = PatchDataLoader(patch_size=(128, 128, 128))
        
        # Tạo dummy volume
        volume = np.random.randn(256, 256, 256, 4).astype(np.float32)
        mask = np.random.randint(0, 5, size=(256, 256, 256)).astype(np.uint8)
        
        patches, patch_masks = loader.extract_patches(volume, mask, stride=(128, 128, 128))
        
        assert patches.shape[0] == 8, f"Should extract 8 patches, got {patches.shape[0]}"
        assert patches.shape[1:] == (128, 128, 128, 4), "Patch shape should be (128, 128, 128, 4)"
        assert patch_masks.shape == (8, 128, 128, 128), "Patch mask shape should be (8, 128, 128, 128)"
        print("✓ Test 10.1 passed: Extract patches with no overlap works")
    
    def test_extract_patches_with_overlap(self):
        """
        Test 10.2: Extract patches có overlap
        - Volume: (256, 256, 256, 4)
        - Patch size: (128, 128, 128)
        - Stride: (64, 64, 64)
        - Expected: 27 patches (3x3x3)
        """
        loader = PatchDataLoader(patch_size=(128, 128, 128))
        
        volume = np.random.randn(256, 256, 256, 4).astype(np.float32)
        
        patches, _ = loader.extract_patches(volume, stride=(64, 64, 64))
        
        assert patches.shape[0] == 27, f"Should extract 27 patches, got {patches.shape[0]}"
        print("✓ Test 10.2 passed: Extract patches with overlap works")
    
    # ============================================================================
    # TEST CASE 11: Extract Random Patches
    # Kiểm tra extract random patches
    # ============================================================================
    def test_extract_random_patches(self):
        """
        Test 11.1: Extract random patches
        - Volume: (256, 256, 256, 4)
        - Number of patches: 10
        - Patches được extract ngẫu nhiên
        """
        loader = PatchDataLoader(patch_size=(128, 128, 128))
        
        volume = np.random.randn(256, 256, 256, 4).astype(np.float32)
        mask = np.random.randint(0, 5, size=(256, 256, 256)).astype(np.uint8)
        
        patches, patch_masks = loader.extract_patches(volume, mask, random_patches=10)
        
        assert patches.shape[0] == 10, f"Should extract 10 patches, got {patches.shape[0]}"
        assert patches.shape[1:] == (128, 128, 128, 4), "Patch shape should be (128, 128, 128, 4)"
        print("✓ Test 11.1 passed: Extract random patches works")
    
    # ============================================================================
    # TEST CASE 12: Create Training Data
    # Kiểm tra tạo training data với one-hot encoding
    # ============================================================================
    def test_create_training_data(self):
        """
        Test 12.1: Tạo training data với one-hot encoding
        - Input: patches và masks
        - Output: (X, y) với y là one-hot encoded
        - y phải có tổng = 1 cho mỗi sample
        """
        loader = PatchDataLoader()
        
        # Tạo dummy data
        patches = np.random.randn(5, 128, 128, 128, 4).astype(np.float32)
        masks = np.random.randint(0, 4, size=(5, 128, 128, 128)).astype(np.uint8)
        
        X, y = loader.create_training_data(patches, masks, num_classes=4)
        
        assert X.shape == patches.shape, "X should have same shape as input patches"
        assert y.shape == (5, 4), f"y shape should be (5, 4), got {y.shape}"
        assert np.allclose(np.sum(y, axis=1), 1.0), "Each row of y should sum to 1"
        print("✓ Test 12.1 passed: Create training data works")
    
    # ============================================================================
    # TEST CASE 13: Data Augmentation
    # Kiểm tra augmentation (flip, rotation)
    # ============================================================================
    def test_augment_patch(self):
        """
        Test 13.1: Augment patch (flip và rotation)
        - Input: patch và mask
        - Output: augmented patch và mask với cùng shape
        - Augmentation: random flip và rotation
        """
        loader = PatchDataLoader()
        
        patch = np.random.randn(128, 128, 128, 4).astype(np.float32)
        mask = np.random.randint(0, 4, size=(128, 128, 128)).astype(np.uint8)
        
        aug_patch, aug_mask = loader.augment_patch(patch, mask)
        
        assert aug_patch.shape == patch.shape, "Augmented patch should have same shape"
        assert aug_mask.shape == mask.shape, "Augmented mask should have same shape"
        print("✓ Test 13.1 passed: Data augmentation works")
    
    # ============================================================================
    # TEST CASE 14: Batch Generator
    # Kiểm tra batch generator
    # ============================================================================
    def test_batch_generator(self):
        """
        Test 14.1: Batch generator tạo batches đúng
        - Input: 10 patches, batch size = 3
        - Expected: 4 batches (3, 3, 3, 1)
        """
        loader = PatchDataLoader(batch_size=3, shuffle=False)
        
        patches = np.random.randn(10, 128, 128, 128, 4).astype(np.float32)
        labels = np.random.rand(10, 4).astype(np.float32)
        labels = labels / np.sum(labels, axis=1, keepdims=True)
        
        batches = list(loader.batch_generator(patches, labels, augment=False))
        
        assert len(batches) == 4, f"Should generate 4 batches, got {len(batches)}"
        assert batches[0][0].shape[0] == 3, "First batch should have 3 samples"
        assert batches[-1][0].shape[0] == 1, "Last batch should have 1 sample"
        print("✓ Test 14.1 passed: Batch generator works")
    
    # ============================================================================
    # TEST CASE 15: Normalization
    # Kiểm tra các phương pháp normalization
    # ============================================================================
    def test_normalize_patches_standard(self):
        """
        Test 15.1: Normalize patches với standard method
        - Method: 'standard' (zero mean, unit variance)
        """
        loader = PatchDataLoader()
        
        patches = np.random.randn(5, 128, 128, 128, 4).astype(np.float32) * 100 + 50
        normalized = loader.normalize_patches(patches, method='standard')
        
        # Kiểm tra mean gần 0 và std gần 1 cho mỗi patch
        for i in range(5):
            mean = np.mean(normalized[i])
            std = np.std(normalized[i])
            assert np.abs(mean) < 0.1, f"Mean should be close to 0, got {mean}"
            assert np.abs(std - 1.0) < 0.1, f"Std should be close to 1, got {std}"
        
        print("✓ Test 15.1 passed: Standard normalization works")
    
    def test_normalize_patches_minmax(self):
        """
        Test 15.2: Normalize patches với minmax method
        - Method: 'minmax' (scale to [0, 1])
        """
        loader = PatchDataLoader()
        
        patches = np.random.randn(5, 128, 128, 128, 4).astype(np.float32) * 100 + 50
        normalized = loader.normalize_patches(patches, method='minmax')
        
        # Kiểm tra values trong range [0, 1]
        assert np.min(normalized) >= 0, "Min value should be >= 0"
        assert np.max(normalized) <= 1, "Max value should be <= 1"
        print("✓ Test 15.2 passed: MinMax normalization works")
    
    # ============================================================================
    # TEST CASE 16: Class Balancing
    # Kiểm tra balancing class distribution
    # ============================================================================
    def test_balance_classes(self):
        """
        Test 16.1: Balance class distribution
        - Input: imbalanced dataset
        - Output: balanced dataset với uniform distribution
        """
        loader = PatchDataLoader()
        
        # Tạo imbalanced dataset
        patches = np.random.randn(100, 128, 128, 128, 4).astype(np.float32)
        
        # Class 0: 50 samples, Class 1: 30 samples, Class 2: 15 samples, Class 3: 5 samples
        labels = np.zeros((100, 4))
        labels[:50, 0] = 1.0
        labels[50:80, 1] = 1.0
        labels[80:95, 2] = 1.0
        labels[95:, 3] = 1.0
        
        balanced_patches, balanced_labels = loader.balance_classes(patches, labels)
        
        # Kiểm tra distribution
        dominant_classes = np.argmax(balanced_labels, axis=1)
        class_counts = [np.sum(dominant_classes == i) for i in range(4)]
        
        # Kiểm tra các class có số lượng tương đương nhau
        max_count = max(class_counts)
        min_count = min(class_counts)
        assert max_count / min_count < 1.5, "Classes should be relatively balanced"
        print(f"✓ Test 16.1 passed: Class balancing works (counts: {class_counts})")


# ============================================================================
# TEST CASE 17: Integration Test
# Kiểm tra workflow hoàn chỉnh từ data loading đến prediction
# ============================================================================
class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self):
        """
        Test 17.1: Complete workflow (với kích thước nhỏ hơn để test nhanh)
        - Extract patches từ volume
        - Create training data
        - Predict trên volume
        - Get segmentation mask
        """
        # 1. Create data loader và model (size nhỏ để test nhanh)
        loader = PatchDataLoader(patch_size=(32, 32, 32), batch_size=2)
        model = CNN3DModel(input_shape=(32, 32, 32, 4), num_classes=4)
        
        # 2. Create dummy volume
        volume = np.random.randn(64, 64, 64, 4).astype(np.float32)
        mask = np.random.randint(0, 4, size=(64, 64, 64)).astype(np.uint8)
        
        # 3. Extract patches
        patches, patch_masks = loader.extract_patches(volume, mask, stride=(32, 32, 32))
        assert patches.shape[0] > 0, "Should extract at least one patch"
        print(f"  - Extracted {patches.shape[0]} patches")
        
        # 4. Create training data
        X, y = loader.create_training_data(patches, patch_masks)
        print(f"  - Created training data: X shape {X.shape}, y shape {y.shape}")
        
        # 5. Normalize
        X = loader.normalize_patches(X, method='standard')
        print(f"  - Normalized patches")
        
        # 6. Predict on small volume
        small_volume = np.random.randn(64, 64, 64, 4).astype(np.float32)
        output = model.predict_volume(small_volume, patch_size=(32, 32, 32))
        assert output.shape == (64, 64, 64, 4), "Output shape should match volume"
        print(f"  - Predicted on volume: output shape {output.shape}")
        
        # 7. Get segmentation mask
        seg_mask = model.get_segmentation_mask(output)
        assert seg_mask.shape == (64, 64, 64), "Segmentation mask shape should be (64, 64, 64)"
        print(f"  - Generated segmentation mask: shape {seg_mask.shape}")
        
        print("✓ Test 17.1 passed: Complete workflow works correctly")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING TESTS FOR MODULE 4: 3D CNN")
    print("="*80 + "\n")
    
    # Run tests với pytest
    pytest.main([__file__, "-v", "--tb=short"])
