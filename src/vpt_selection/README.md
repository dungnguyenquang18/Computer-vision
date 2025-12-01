# Module 3: Lựa Chọn Đặc Trưng VPT (VPT Feature Selection)

## Mô tả chức năng

Module này sử dụng cấu trúc Vantage Point Tree (VPT) để lựa chọn và tinh chọn các đặc trưng quan trọng nhất, giúp giảm chiều dữ liệu và loại bỏ nhiễu.

## Input

- **Dữ liệu:** Tensor đặc trưng GLCM 20 kênh
- **Kích thước:** `240 × 240 × 160 × 20`

## Thuật toán

VPT là cấu trúc dữ liệu dạng cây metric dùng cho tìm kiếm láng giềng gần nhất (Nearest Neighbor Search) hiệu quả trong không gian nhiều chiều.

### Vai trò trong pipeline:

- **Feature Selection (Lựa chọn đặc trưng)**
- **Tiền phân loại (Pre-classification)**

### Các bước thực hiện:

1. **Xây dựng cây VPT:**

   - Các vector đặc trưng 20 chiều của từng voxel/patch được đưa vào cây
   - Chọn các vantage points (điểm có lợi) để phân chia không gian

2. **Truy vấn (Indexing):**

   - Tìm kiếm các vector đặc trưng mẫu (đã biết nhãn u/không u) gần nhất với vector hiện tại
   - Sử dụng khoảng cách metric (Euclidean, Manhattan, etc.)

3. **Lọc đặc trưng:**
   - Loại bỏ các đặc trưng nhiễu
   - Loại bỏ đặc trưng không đóng góp vào phân tách u/mô lành
   - Xác định "highlight vectors" (vector đại diện)

## Output

- **Kích thước:** `240 × 240 × 160 × N` (N ≤ 20)
- **Đặc điểm:**
  - Giảm số kênh từ 20 xuống N kênh quan trọng nhất
  - Có thể bao gồm bản đồ xác suất tiên nghiệm (prior probability maps)
  - Giảm tải tính toán và tăng độ chính xác cho mạng nơ-ron

## Cấu trúc file sẽ implement

```
03-vpt-selection/
├── README.md (file này)
├── vpt_selector.py (class VPTSelector)
└── __init__.py
```

## Class chính: `VPTSelector`

### Methods:

- `__init__(n_features=10, distance_metric='euclidean')`: Khởi tạo với số đặc trưng giữ lại
- `build_tree(feature_vectors, labels)`: Xây dựng cây VPT từ dữ liệu training
- `query_nearest(vector, k=5)`: Tìm k láng giềng gần nhất
- `select_features(volume)`: Lựa chọn đặc trưng quan trọng
- `compute_prior_maps(volume)`: Tính bản đồ xác suất tiên nghiệm
