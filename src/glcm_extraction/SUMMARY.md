# Module 02: GLCM Extraction - Hoàn Thành ✅

## Tóm tắt

Module trích xuất đặc trưng GLCM đã được implement đầy đủ với comprehensive test suite.

## Files đã tạo

### 1. `glcm_extractor.py` - Module chính

**Class:** `GLCMExtractor`

**Chức năng chính:**

- Tính toán GLCM (Grey Level Co-occurrence Matrix)
- Trích xuất 5 đặc trưng Haralick cho mỗi kênh
- Transform: 4 kênh input → 20 kênh features (4 × 5)

**Methods:**

- `__init__(window_size=5, distances=[1], angles=[...], levels=32)` - Khởi tạo
- `_quantize_image(image)` - Quantize ảnh về số mức xám cố định
- `compute_glcm_2d(patch_2d)` - Tính GLCM cho patch 2D
- `extract_haralick_features_2d(glcm)` - Trích xuất 5 features từ GLCM
- `extract_features_from_slice(slice_2d)` - Xử lý slice 2D với sliding window
- `extract_features_channel(channel_volume)` - Xử lý 1 kênh 3D
- `extract_features(volume)` - Pipeline đầy đủ cho 4D volume
- `extract_features_fast(volume, stride)` - Fast mode với stride

**5 Đặc trưng Haralick:**

1. **Contrast** (Differentiation - Độ tương phản)
2. **Dissimilarity** (Divergence - Phân kỳ)
3. **Homogeneity** (Đồng nhất)
4. **Energy** (Năng lượng)
5. **Correlation** (Relationship - Quan hệ)

### 2. `test_glcm_extractor.py` - Comprehensive test suite

**10 Test Cases:**

#### ✅ Test Case 1: Initialization

- Default parameters
- Custom parameters
- Feature names verification

#### ✅ Test Case 2: Image Quantization

- Normal quantization to gray levels
- Constant image handling
- Quantization level verification

#### ✅ Test Case 3: GLCM Computation

- Basic GLCM computation for 2D patches
- GLCM shape and normalization
- Empty patch handling

#### ✅ Test Case 4: Haralick Features

- Extract 5 features from GLCM
- Feature value ranges (homogeneity, energy in [0,1])
- None GLCM handling

#### ✅ Test Case 5: 2D Slice Processing

- Sliding window on 2D slices
- Boundary handling (edges)
- Structured patterns (checkerboard)

#### ✅ Test Case 6: 3D Channel Processing

- Single channel 3D volume
- Slice independence
- Feature statistics

#### ✅ Test Case 7: Full 4D Volume

- 4 channels → 20 features transformation
- Channel mapping verification
- Different channel counts (2→10, 4→20)

#### ✅ Test Case 8: Fast Mode

- Fast extraction with stride
- Output size reduction
- Different stride values (1, 2, 4)

#### ✅ Test Case 9: Edge Cases

- Invalid shapes (reject 3D)
- Very small volumes
- Extreme values (1e6 range)
- All-zeros volume

#### ✅ Test Case 10: Realistic Scenario

- Scaled BraTS-like volume (60×60×40×4)
- Integration with preprocessed data
- Feature statistics validation

### 3. `__init__.py` - Package initialization

Export `GLCMExtractor` class.

### 4. `SUMMARY.md` - Tài liệu này

## Cách sử dụng

### Basic usage:

```python
from glcm_extractor import GLCMExtractor
import numpy as np

# Khởi tạo
extractor = GLCMExtractor(
    window_size=5,      # Cửa sổ 5×5×5
    distances=[1],      # Khoảng cách 1 pixel
    levels=32           # 32 mức xám
)

# Xử lý volume (từ preprocessing: 240×240×160×4)
volume = np.random.randn(240, 240, 160, 4).astype(np.float32)

# Extract features: 4 channels → 20 features
features = extractor.extract_features(volume)
print(features.shape)  # Output: (240, 240, 160, 20)
```

### Fast mode cho testing:

```python
# Fast mode với stride=2 (nhanh hơn ~4x)
features_fast = extractor.extract_features_fast(volume, stride=2)
# Output shape sẽ nhỏ hơn do stride
```

### Integration với Preprocessing:

```python
# Pipeline đầy đủ
from preprocessing.preprocessor import Preprocessor
from glcm_extractor import GLCMExtractor

# Step 1: Preprocessing (240×240×155×4 → 240×240×160×4)
preprocessor = Preprocessor()
preprocessed = preprocessor.preprocess(raw_volume)

# Step 2: GLCM Feature Extraction (240×240×160×4 → 240×240×160×20)
extractor = GLCMExtractor()
features = extractor.extract_features(preprocessed)
```

## Chạy tests

```bash
cd 02-glcm-extraction
python test_glcm_extractor.py
```

## Dependencies

- **Bắt buộc:**
  - `numpy`
  - `scipy` (cho uniform_filter)
  - `scikit-image` (cho graycomatrix, graycoprops)

## Đặc điểm kỹ thuật

### Input/Output:

- ✅ Input: 240×240×160×4 (từ preprocessing)
- ✅ Output: 240×240×160×20 (4 channels × 5 features)
- ✅ Transform: C channels → C×5 feature channels

### GLCM Parameters:

- **Window size:** 3, 5, 7 (sliding window size)
- **Distances:** [1] (pixel distance cho co-occurrence)
- **Angles:** 4 directions (0°, 45°, 90°, 135°)
- **Gray levels:** 8, 16, 32, 64 (quantization levels)

### Performance:

- **Normal mode:** ~5-10s cho volume nhỏ (20×20×5×4)
- **Fast mode (stride=2):** ~2-3s (giảm ~50% time)
- **Stride=4:** ~1s (giảm ~80% time)

### Features Output:

Mỗi feature channel chứa:

- **Contrast:** Độ tương phản cục bộ
- **Dissimilarity:** Độ phân kỳ texture
- **Homogeneity:** Độ đồng nhất (0-1)
- **Energy:** Năng lượng texture (0-1)
- **Correlation:** Tương quan spatial

## Output mẫu từ test:

```
============================================================
[GLCMExtractor] Starting feature extraction...
============================================================
Input shape: (20, 20, 5, 4)
Output will be: (20, 20, 5, 20)

[GLCMExtractor] Processing channel 1/4...
  [GLCMExtractor] Processing 5 slices... Done!
    contrast: mean=2.8063, std=2.1464
    dissimilarity: mean=1.1772, std=0.6910
    homogeneity: mean=0.3738, std=0.2040
    energy: mean=0.3359, std=0.1701
    correlation: mean=-0.1311, std=0.1464

[GLCMExtractor] Feature extraction completed!
Final shape: (20, 20, 5, 20)
============================================================

🎉 ALL TESTS PASSED! Module is ready for use.
```

## Technical Notes

### Sliding Window Approach:

- Cửa sổ trượt quét qua mỗi voxel
- Tính GLCM cho mỗi patch cục bộ
- Edges được xử lý bằng cách không compute (giữ zero)

### Quantization:

- Normalize ảnh về [0, 1]
- Quantize về N mức xám (8, 16, 32, 64)
- Giúp GLCM computation ổn định hơn

### Multi-directional:

- Tính GLCM theo 4 hướng (0°, 45°, 90°, 135°)
- Average features across directions
- Rotation-invariant features

## Pipeline Flow:

```
Input: (H, W, D, C=4)
    ↓
For each channel:
    For each slice:
        Sliding window → GLCM → 5 features
    ↓
Stack all features
    ↓
Output: (H, W, D, C×5=20)
```

## Next Steps

Tiếp theo: Module 03 - VPT Feature Selection (20 → N channels)
