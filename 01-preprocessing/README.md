# Module 1: Tiền Xử Lý Ảnh MRI (MRI Preprocessing)

## Mô tả chức năng

Module này thực hiện tiền xử lý ảnh MRI đầu vào, bao gồm zero-padding và chuẩn hóa Z-score.

## Input

- **Dữ liệu:** Ảnh MRI đa phương thức (Multimodal MRI) từ bộ dữ liệu BraTS
- **Số kênh:** 4 kênh
  - T1-weighted (T1)
  - T1-weighted contrast-enhanced (T1ce)
  - T2-weighted (T2)
  - Fluid-Attenuated Inversion Recovery (FLAIR)
- **Kích thước:** `240 × 240 × 155 × 4` voxels
- **Định dạng:** NIfTI (`.nii` hoặc `.nii.gz`)

## Thuật toán

1. **Zero-padding:** Thêm các voxel giá trị 0 vào chiều sâu (depth) từ 155 lên 160 để đảm bảo chia hết cho các tầng pooling trong mạng Deep Learning
2. **Normalization (Z-score):** Chuẩn hóa từng kênh về phân phối chuẩn
   - Công thức: `z = (x - μ) / σ`
   - Áp dụng độc lập cho mỗi kênh

## Output

- **Kích thước:** `240 × 240 × 160 × 4`
- **Đặc điểm:** Dữ liệu đã được chuẩn hóa, sẵn sàng cho bước trích xuất đặc trưng

## Cấu trúc file sẽ implement

```
01-preprocessing/
├── README.md (file này)
├── preprocessor.py (class Preprocessor)
└── __init__.py
```

## Class chính: `Preprocessor`

### Methods:

- `__init__(target_shape=(240, 240, 160, 4))`: Khởi tạo với kích thước đích
- `load_nifti(filepath)`: Đọc file NIfTI
- `pad_volume(volume)`: Padding chiều depth từ 155 → 160
- `normalize_zscore(volume)`: Chuẩn hóa Z-score cho từng kênh
- `preprocess(volume)`: Pipeline đầy đủ
