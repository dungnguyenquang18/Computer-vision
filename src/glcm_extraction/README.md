# Module 2: Trích Xuất Đặc Trưng GLCM (GLCM Feature Extraction)

## Mô tả chức năng

Module này trích xuất các đặc trưng kết cấu (texture features) từ ảnh MRI sử dụng ma trận đồng xuất hiện mức xám (Grey Level Co-occurrence Matrix - GLCM).

## Input

- **Dữ liệu:** Khối ảnh MRI 4 kênh đã qua tiền xử lý
- **Kích thước:** `240 × 240 × 160 × 4`

## Thuật toán

GLCM được sử dụng để khai thác thông tin kết cấu mà mắt thường hoặc cường độ pixel đơn lẻ khó phát hiện.

### Các bước thực hiện:

1. **Quét cửa sổ (Sliding Window):**

   - Cửa sổ trượt quét qua từng voxel của khối ảnh 3D
   - Kích thước cửa sổ có thể là 3×3×3 hoặc 5×5×5

2. **Tính toán quan hệ không gian:**

   - Tại mỗi vị trí, tính tần suất xuất hiện của các cặp pixel
   - Theo các hướng: 0°, 45°, 90°, 135°
   - Với khoảng cách (distance) nhất định

3. **Trích xuất 5 chỉ số Haralick cho mỗi kênh:**
   - **Relationship (Quan hệ):** Correlation giữa các pixel
   - **Divergence (Phân kỳ):** Mức độ khác biệt
   - **Homogeneity (Đồng nhất):** Độ đồng đều của kết cấu
   - **Differentiation (Vi phân/Độ tương phản):** Contrast
   - **Energy (Năng lượng):** Angular Second Moment

## Output

- **Kích thước:** `240 × 240 × 160 × 20`
- **Giải thích:**
  - 4 kênh gốc × 5 đặc trưng = 20 kênh đầu ra
  - Giá trị tại mỗi voxel là giá trị đặc trưng kết cấu (không còn là cường độ sáng)

## Cấu trúc file sẽ implement

```
02-glcm-extraction/
├── README.md (file này)
├── glcm_extractor.py (class GLCMExtractor)
└── __init__.py
```

## Class chính: `GLCMExtractor`

### Methods:

- `__init__(window_size=3, distances=[1], angles=[0, 45, 90, 135])`: Khởi tạo tham số GLCM
- `compute_glcm(patch)`: Tính ma trận GLCM cho một patch
- `extract_haralick_features(glcm_matrix)`: Tính 5 đặc trưng Haralick
- `extract_features(volume)`: Trích xuất đặc trưng cho toàn bộ volume
