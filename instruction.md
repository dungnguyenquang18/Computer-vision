# Pipeline Phân Đoạn U Não Tự Động (Automated Brain Tumor Segmentation Pipeline)

Tài liệu này mô tả chi tiết quy trình xử lý (pipeline) từ ảnh MRI thô đến kết quả phân đoạn cuối cùng, dựa trên phương pháp đề xuất kết hợp trích xuất đặc trưng (GLCM, VPT) và Deep Learning (3D CNN, U-Net).

---

## 1. Ảnh MRI Đầu vào (Input MRI)

### Mô tả Input
* **Dữ liệu:** Ảnh cộng hưởng từ đa phương thức (Multimodal MRI) thuộc bộ dữ liệu BraTS.
* **Số lượng kênh (Channels):** 4 kênh bao gồm:
    1.  T1-weighted (T1)
    2.  T1-weighted contrast-enhanced (T1ce)
    3.  T2-weighted (T2)
    4.  [cite_start]Fluid-Attenuated Inversion Recovery (FLAIR)[cite: 35].
* **Kích thước gốc:** Khối 3D kích thước $240 \times 240 \times 155$ voxels.
* **Định dạng:** NIfTI (`.nii` hoặc `.nii.gz`).

### Mô tả Output (của bước này)
* **Kích thước:** $240 \times 240 \times 160 \times 4$.
* **Giải thích:** Trước khi đưa vào pipeline, dữ liệu thường được tiền xử lý:
    * [cite_start]**Zero-padding:** Thêm các voxel giá trị 0 vào chiều sâu (depth) từ 155 lên 160 để đảm bảo chia hết cho các tầng pooling trong mạng Deep Learning phía sau[cite: 289].
    * [cite_start]**Normalization:** Chuẩn hóa Z-score để đưa giá trị cường độ điểm ảnh về cùng phân phối[cite: 302].

---

## 2. Trích xuất đặc trưng GLCM (Grey Level Co-occurrence Matrix)

### Mô tả Input
* **Dữ liệu:** Khối ảnh MRI 4 kênh từ bước 1.
* **Kích thước:** $240 \times 240 \times 160 \times 4$.

### Mô tả Thuật toán
[cite_start]GLCM (Ma trận đồng xuất hiện mức xám) được sử dụng để khai thác thông tin kết cấu (texture) mà mắt thường hoặc cường độ pixel đơn lẻ khó phát hiện[cite: 20].
1.  **Quét cửa sổ:** Một cửa sổ trượt (sliding window) quét qua từng voxel của khối ảnh 3D.
2.  [cite_start]**Tính toán quan hệ không gian:** Tại mỗi vị trí, tính tần suất xuất hiện của các cặp pixel theo các hướng và khoảng cách nhất định (ví dụ: $0^\circ, 45^\circ, 90^\circ, 135^\circ$)[cite: 359].
3.  **Trích xuất chỉ số Haralick:** Từ ma trận tần suất, tính toán 5 đặc trưng thống kê cốt lõi cho mỗi kênh ảnh:
    * Relationship (Quan hệ)
    * Divergence (Phân kỳ)
    * Homogeneity (Đồng nhất)
    * Differentiation (Vi phân/Độ tương phản)
    * [cite_start]Energy (Năng lượng)[cite: 359].

### Mô tả Output
* **Dữ liệu:** Tensor đặc trưng kết cấu.
* **Kích thước:** $240 \times 240 \times 160 \times 20$.
* **Giải thích thay đổi chiều:**
    * Input có 4 kênh gốc.
    * Với mỗi kênh gốc, ta trích xuất được 5 bản đồ đặc trưng GLCM mới.
    * Tổng số kênh đầu ra = $4 \text{ (kênh gốc)} \times 5 \text{ (đặc trưng)} = 20 \text{ kênh}$.
    * Giá trị tại mỗi voxel lúc này không còn là cường độ sáng mà là giá trị đặc trưng kết cấu.

---

## 3. Lựa chọn đặc trưng VPT (Vantage Point Tree)

### Mô tả Input
* **Dữ liệu:** Tensor đặc trưng GLCM 20 kênh từ bước 2.
* **Kích thước:** $240 \times 240 \times 160 \times 20$.

### Mô tả Thuật toán
[cite_start]VPT là một cấu trúc dữ liệu dạng cây metric được sử dụng để tìm kiếm láng giềng gần nhất (Nearest Neighbor Search) hiệu quả trong không gian nhiều chiều[cite: 151]. [cite_start]Trong pipeline này, VPT đóng vai trò **Feature Selection (Lựa chọn đặc trưng)** và tiền phân loại[cite: 375].
1.  **Xây dựng cây:** Các vector đặc trưng 20 chiều của từng voxel (hoặc patch) được đưa vào cây VPT.
2.  **Truy vấn (Indexing):** Tìm kiếm các vector đặc trưng mẫu (đã biết nhãn u/không u) gần nhất với vector hiện tại.
3.  **Lọc đặc trưng:** Loại bỏ các đặc trưng nhiễu hoặc không đóng góp vào việc phân tách giữa khối u và mô lành. [cite_start]VPT giúp xác định các vector đặc trưng nào mang tính đại diện nhất ("highlight vectors")[cite: 379].

### Mô tả Output
* **Dữ liệu:** Tensor đặc trưng đã được tinh chọn (Refined Feature Tensor).
* **Kích thước:** $240 \times 240 \times 160 \times N$ (trong đó $N \le 20$).
* **Giải thích thay đổi:**
    * Mục đích của VPT ở đây là giảm chiều dữ liệu hoặc làm sạch dữ liệu trước khi đưa vào mạng nơ-ron để giảm tải tính toán và tăng độ chính xác.
    * Đầu ra vẫn giữ nguyên không gian không gian 3D, nhưng số lượng kênh đặc trưng ($N$) có thể giảm xuống hoặc được thay thế bằng các bản đồ xác suất tiên nghiệm (prior probability maps) từ VPT.

---

## 4. Mạng 3D CNN (Deep Convolutional Neural Network)

### Mô tả Input
* **Dữ liệu:** Các vùng ảnh (Patches) được cắt từ Tensor đặc trưng sau bước VPT/GLCM.
* [cite_start]**Kích thước Input:** $128 \times 128 \times 128 \times N$ (Batch size = 2)[cite: 286].
* **Lý do thay đổi Input:** Không thể đưa toàn bộ khối $240 \times 240 \times 160$ vào mạng cùng lúc do giới hạn bộ nhớ GPU. Dữ liệu phải được cắt thành các khối nhỏ (patches) chồng lấn nhau.

### Mô tả Thuật toán
[cite_start]Mô hình này tập trung vào việc phân loại voxel dựa trên ngữ cảnh cục bộ[cite: 248].
1.  [cite_start]**Convolution Layers:** Sử dụng các kernel 3D ($3 \times 3 \times 3$) để trích xuất đặc trưng sâu hơn[cite: 233].
2.  [cite_start]**Pooling Layers:** Max-pooling để giảm kích thước không gian và giữ lại đặc trưng nổi bật[cite: 239].
3.  **Fully Connected Layers:** Ở cuối mạng để phân loại từng patch hoặc voxel vào các lớp (U, Phù nề, Hoại tử, Nền).
4.  **Activation:** Sử dụng ReLU và Softmax cho lớp cuối cùng.

### Mô tả Output
* **Kích thước:** Bản đồ xác suất cục bộ cho từng patch ($128 \times 128 \times 128 \times \text{Số lớp}$).

---

## 5. Mạng 3D U-Net

### Mô tả Input
* **Dữ liệu:** Tương tự như CNN, đầu vào là các patches từ dữ liệu đã qua xử lý.
* [cite_start]**Kích thước Input:** $128 \times 128 \times 128 \times N$[cite: 304].

### Mô tả Thuật toán
[cite_start]U-Net có kiến trúc dạng chữ U chuyên dụng cho phân đoạn ngữ nghĩa[cite: 334]:
1.  **Contracting Path (Encoder):** Giống CNN, dùng để trích xuất ngữ cảnh và giảm kích thước không gian.
2.  [cite_start]**Expanding Path (Decoder):** Dùng các lớp Up-sampling và Transposed Convolution để khôi phục lại kích thước ảnh gốc[cite: 295].
3.  [cite_start]**Skip Connections:** Nối (concatenate) các bản đồ đặc trưng từ Encoder sang Decoder tương ứng để giữ lại thông tin chi tiết về vị trí không gian (spatial information) bị mất trong quá trình pooling[cite: 293].

### Mô tả Output
* **Kích thước:** Bản đồ phân đoạn cho từng patch ($128 \times 128 \times 128 \times \text{Số lớp}$).
* **Đặc điểm:** U-Net thường cho kết quả chi tiết hơn ở các đường biên khối u so với CNN thuần túy.

---

## 6. Dung hợp (Ensemble & Fusion)

### Mô tả Input
* **Dữ liệu:**
    1.  Bản đồ xác suất từ 3D CNN (đã được ghép lại từ các patches thành khối lớn $240 \times 240 \times 160$).
    2.  Bản đồ xác suất từ 3D U-Net (đã được ghép lại từ các patches thành khối lớn $240 \times 240 \times 160$).

### Mô tả Thuật toán
[cite_start]Bước này kết hợp sức mạnh của cả hai mạng để đưa ra quyết định cuối cùng[cite: 22].
1.  **Reconstruction:** Ghép các patches đầu ra từ bước 4 và 5 lại đúng vị trí ban đầu trong không gian $240 \times 240 \times 160$.
2.  **Ensemble Strategy (Variable Assembly):**
    * Tại mỗi voxel $(x, y, z)$, lấy vector xác suất từ CNN ($P_{cnn}$) và từ U-Net ($P_{unet}$).
    * [cite_start]Thực hiện tính trung bình (Averaging) hoặc bỏ phiếu đa số (Majority Voting) có trọng số[cite: 618].
    * Công thức đơn giản hóa: $P_{final} = \alpha \cdot P_{cnn} + \beta \cdot P_{unet}$.
3.  **Argmax:** Chọn lớp có xác suất cao nhất làm nhãn cuối cùng cho voxel đó.
4.  **Post-processing:** Cắt bỏ phần padding (từ 160 về 155) để trả về kích thước gốc.

### Mô tả Output (Final Output)
* **Dữ liệu:** Mask phân đoạn khối u (Segmentation Mask).
* **Kích thước:** $240 \times 240 \times 155$.
* **Giá trị:** Các số nguyên đại diện cho nhãn (0: Nền, 1: Hoại tử, 2: Phù nề, 4: U bắt thuốc).