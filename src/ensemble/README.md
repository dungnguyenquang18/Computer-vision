# Module 6: Dung Hợp (Ensemble & Fusion)

## Mô tả chức năng

Module này kết hợp kết quả từ cả 3D CNN và 3D U-Net để đưa ra quyết định phân đoạn cuối cùng, tận dụng điểm mạnh của cả hai mô hình.

## Input

Hai bản đồ xác suất đã được ghép từ patches:

1. **Từ 3D CNN:** `240 × 240 × 160 × num_classes`
2. **Từ 3D U-Net:** `240 × 240 × 160 × num_classes`

## Thuật toán

Kết hợp sức mạnh của cả hai mạng để tăng độ chính xác và ổn định.

### Các bước thực hiện:

#### 1. **Reconstruction (Tái tạo):**

- Ghép các patches đầu ra từ CNN và U-Net
- Đưa về đúng vị trí ban đầu trong không gian `240 × 240 × 160`
- Xử lý vùng chồng lấn (overlapping regions) bằng averaging

#### 2. **Ensemble Strategy (Chiến lược kết hợp):**

##### **Option A: Weighted Averaging (Trung bình có trọng số)**

```
P_final(x,y,z) = α · P_cnn(x,y,z) + β · P_unet(x,y,z)
```

- α + β = 1
- Thường α = 0.4, β = 0.6 (U-Net tốt hơn ở biên)

##### **Option B: Majority Voting (Bỏ phiếu đa số)**

- Lấy class có xác suất cao nhất từ mỗi model
- Nếu trùng nhau → chọn class đó
- Nếu khác nhau → chọn theo model có confidence cao hơn

##### **Option C: Hybrid Approach**

- Averaging cho vùng không chắc chắn (low confidence)
- Hard voting cho vùng chắc chắn (high confidence)

#### 3. **Argmax:**

- Tại mỗi voxel, chọn lớp có xác suất cao nhất
- Tạo mask phân đoạn cuối cùng

#### 4. **Post-processing:**

- **Unpadding:** Cắt bỏ padding (từ 160 về 155) để về kích thước gốc
- **Morphological operations:**
  - Loại bỏ các vùng nhỏ nhiễu (connected component analysis)
  - Làm mịn ranh giới (optional)
- **Consistency check:** Đảm bảo logic (ví dụ: necrotic core phải nằm trong tumor)

## Output (Final Output)

- **Kích thước:** `240 × 240 × 155`
- **Dạng:** Segmentation mask (mask phân đoạn khối u)
- **Giá trị:**
  - 0: Nền (Background)
  - 1: Hoại tử (Necrotic core)
  - 2: Phù nề (Edema)
  - 4: Khối u bắt thuốc (Enhancing tumor)

## Cấu trúc file sẽ implement

```
06-ensemble/
├── README.md (file này)
├── ensemble.py (class EnsembleModel)
├── postprocessing.py (các hàm post-processing)
└── __init__.py
```

## Class chính: `EnsembleModel`

### Methods:

- `__init__(alpha=0.4, beta=0.6, strategy='weighted')`: Khởi tạo với trọng số và chiến lược
- `reconstruct_from_patches(patches, positions)`: Ghép patches thành volume
- `weighted_average(prob_cnn, prob_unet)`: Trung bình có trọng số
- `majority_voting(prob_cnn, prob_unet)`: Bỏ phiếu đa số
- `ensemble(prob_cnn, prob_unet)`: Áp dụng chiến lược ensemble
- `postprocess(mask)`: Post-processing (unpad, morphology)
- `predict(volume_cnn_patches, volume_unet_patches)`: Pipeline đầy đủ
