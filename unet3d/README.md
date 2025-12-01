# Module 5: Mạng 3D U-Net (3D U-Net Architecture)

## Mô tả chức năng

Module này implement kiến trúc U-Net 3D chuyên dụng cho phân đoạn ngữ nghĩa (semantic segmentation), tập trung vào việc giữ lại thông tin chi tiết về vị trí không gian.

## Input

- **Dữ liệu:** Patches từ tensor đặc trưng (tương tự CNN)
- **Kích thước:** `128 × 128 × 128 × N`
- **Batch size:** 2

## Thuật toán

U-Net có kiến trúc dạng chữ U với hai nhánh chính: Encoder và Decoder.

### Kiến trúc mạng:

#### 1. **Contracting Path (Encoder - Nhánh thu hẹp):**

- Các tầng:
  - Conv3D + ReLU
  - Conv3D + ReLU
  - Max Pooling 3D
- Số filters tăng dần: 32 → 64 → 128 → 256

#### 2. **Expanding Path (Decoder - Nhánh mở rộng):**

- Up-sampling để khôi phục kích thước ảnh gốc
- Transposed Convolution (Conv3DTranspose)
- Các tầng:
  - Up-conv 3D
  - Concatenate với skip connection
  - Conv3D + ReLU
  - Conv3D + ReLU
- Số filters giảm dần: 256 → 128 → 64 → 32

#### 3. **Skip Connections (Kết nối tắt):**

- Nối (concatenate) feature maps từ Encoder sang Decoder tương ứng
- Giữ lại thông tin chi tiết về vị trí không gian (spatial information)
- Giúp khôi phục ranh giới chính xác của khối u

#### 4. **Output Layer:**

- Conv3D 1×1×1 với num_classes filters
- Softmax activation

### Điểm mạnh so với CNN thuần:

- Kết quả chi tiết hơn ở các đường biên khối u
- Phân đoạn chính xác hơn nhờ skip connections
- Phù hợp cho dense prediction (dự đoán từng pixel/voxel)

## Output

- **Kích thước:** `128 × 128 × 128 × num_classes`
- **Dạng:** Bản đồ phân đoạn chi tiết cho từng patch
- **Đặc điểm:** Ranh giới khối u sắc nét, chi tiết hơn CNN

## Cấu trúc file sẽ implement

```
05-unet3d/
├── README.md (file này)
├── unet3d_model.py (class UNet3DModel)
├── layers.py (các custom layers nếu cần)
└── __init__.py
```

## Class chính: `UNet3DModel`

### Methods:

- `__init__(input_shape, num_classes=4, base_filters=32)`: Khởi tạo kiến trúc U-Net
- `build_encoder()`: Xây dựng contracting path
- `build_decoder()`: Xây dựng expanding path
- `build_model()`: Kết hợp encoder-decoder với skip connections
- `train(train_data, val_data, epochs=100)`: Huấn luyện mô hình
- `predict_patch(patch)`: Dự đoán cho một patch
- `predict_volume(volume)`: Dự đoán cho toàn bộ volume
