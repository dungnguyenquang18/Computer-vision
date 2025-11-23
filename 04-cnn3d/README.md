# Module 4: Mạng 3D CNN (3D Convolutional Neural Network)

## Mô tả chức năng

Module này implement mạng nơ-ron tích chập 3D để phân loại voxel dựa trên ngữ cảnh cục bộ.

## Input

- **Dữ liệu:** Patches được cắt từ tensor đặc trưng sau VPT
- **Kích thước:** `128 × 128 × 128 × N`
- **Batch size:** 2
- **Lý do patches:** Không thể đưa toàn bộ khối `240 × 240 × 160` vào mạng cùng lúc do giới hạn bộ nhớ GPU

## Thuật toán

Mô hình tập trung vào phân loại voxel dựa trên ngữ cảnh cục bộ.

### Kiến trúc mạng:

1. **Convolution Layers (Tầng tích chập):**

   - Sử dụng kernel 3D: `3 × 3 × 3`
   - Trích xuất đặc trưng sâu hơn từ volume 3D
   - Số filters tăng dần qua các tầng: 32 → 64 → 128

2. **Pooling Layers (Tầng gộp):**

   - Max-pooling 3D
   - Giảm kích thước không gian
   - Giữ lại đặc trưng nổi bật nhất

3. **Fully Connected Layers (Tầng kết nối đầy đủ):**

   - Flatten output từ conv layers
   - Dense layers để phân loại
   - Dropout để tránh overfitting

4. **Activation Functions:**
   - **ReLU:** Cho các tầng ẩn
   - **Softmax:** Cho lớp đầu ra (phân loại nhiều lớp)

### Các lớp phân đoạn:

- 0: Nền (Background)
- 1: Hoại tử (Necrotic core)
- 2: Phù nề (Edema)
- 4: Khối u bắt thuốc (Enhancing tumor)

## Output

- **Kích thước:** `128 × 128 × 128 × num_classes`
- **Dạng:** Bản đồ xác suất cục bộ cho từng patch
- **Giá trị:** Xác suất thuộc mỗi lớp tại mỗi voxel

## Cấu trúc file sẽ implement

```
04-cnn3d/
├── README.md (file này)
├── cnn3d_model.py (class CNN3DModel)
├── data_loader.py (class PatchDataLoader)
└── __init__.py
```

## Class chính: `CNN3DModel`

### Methods:

- `__init__(input_shape, num_classes=4)`: Khởi tạo kiến trúc mạng
- `build_model()`: Xây dựng các tầng mạng
- `train(train_data, val_data, epochs=100)`: Huấn luyện mô hình
- `predict_patch(patch)`: Dự đoán cho một patch
- `predict_volume(volume)`: Dự đoán cho toàn bộ volume (chia patches)
