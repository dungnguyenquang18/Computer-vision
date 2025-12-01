import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3DModel(nn.Module):
    """
    Mô hình 3D CNN để phân loại voxel (segmentation) dựa trên ngữ cảnh cục bộ.
    Sử dụng kiến trúc Conv3D -> MaxPool3D -> Flatten -> FC Layers.
    """
    def __init__(self, input_channels, num_classes=4):
        """
        Khởi tạo kiến trúc mạng.
        
        :param input_channels: Số kênh đầu vào (N trong 128x128x128xN), sau VPT.
        :param num_classes: Số lượng lớp phân đoạn (4: Necrotic, Edema, Enhancing, Background).
        """
        super(CNN3DModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Xây dựng các tầng mạng
        self.build_model()
        
    def build_model(self):
        """
        Xây dựng các tầng Convolutional, Pooling và Fully Connected.
        ---
        Kiến trúc Convolutional:
        1. 32 filters
        2. 64 filters
        3. 128 filters
        """
        
        # 1. Convolutional and Pooling Layers (Trích xuất đặc trưng)
        
        # Tầng 1: 32 filters
        self.conv1 = nn.Conv3d(self.input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2) # Giảm kích thước không gian / 2
        
        # Tầng 2: 64 filters
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) # Giảm kích thước không gian / 4
        
        # Tầng 3: 128 filters
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2) # Giảm kích thước không gian / 8
        
        # 2. Fully Connected Layers (Phân loại)
        
        # Kích thước đầu ra sau 3 lần MaxPool3D (Patch 128x128x128):
        # 128 / 2 / 2 / 2 = 16
        # Số features sau khi Flatten: 128 * 16 * 16 * 16 = 524,288
        
        # Kích thước Flatten rất lớn, cần giảm bớt hoặc tính toán động
        
        # Tính toán kích thước đầu vào động cho tầng FC đầu tiên
        self.fc_input_features = 128 * (128 // 8) * (128 // 8) * (128 // 8)
        
        # Tầng FC 1
        self.fc1 = nn.Linear(self.fc_input_features, 512)
        self.dropout1 = nn.Dropout(p=0.5) # Dropout để tránh overfitting
        
        # Tầng FC 2 (Output)
        # Sử dụng nn.Conv3d 1x1x1 ở cuối thay vì FC để giữ lại cấu trúc không gian 3D,
        # giúp mô hình phân loại từng voxel.
        # Tuy nhiên, kiến trúc bạn mô tả là CNN -> FC -> Output.
        # Vì bạn nói "phân loại voxel dựa trên ngữ cảnh cục bộ" và output là 128x128x128xnum_classes
        # (Map xác suất), kiến trúc FC truyền thống KHÔNG phù hợp cho Voxel-wise Segmentation.
        # Tôi sẽ điều chỉnh lại kiến trúc để KHÔNG sử dụng Flatten và FC layers,
        # thay vào đó sử dụng kiến trúc Fully Convolutional (FCN) như U-Net:
        
        # --- Kiến trúc được điều chỉnh để phù hợp với Output 128x128x128 ---
        
        # Sử dụng 3D Transposed Conv (Upsampling) để khôi phục kích thước không gian
        
        # Upsampling Tầng 1 (16x16x16 -> 32x32x32)
        self.up_conv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) 
        
        # Upsampling Tầng 2 (32x32x32 -> 64x64x64)
        self.up_conv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        
        # Upsampling Tầng 3 (64x64x64 -> 128x128x128)
        self.up_conv3 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2) 
        
        # Final Output Layer: Conv3D 1x1x1 để ánh xạ số kênh cuối cùng thành số lớp
        self.final_conv = nn.Conv3d(32, self.num_classes, kernel_size=1)
        
        # Không cần Softmax trong mô hình nếu sử dụng nn.CrossEntropyLoss
        
    def forward(self, x):
        """
        Thực hiện quá trình truyền xuôi qua mạng.
        
        :param x: Đầu vào Patch có kích thước [Batch, Channels, D, H, W]
        """
        # 1. Contracting Path (Downsampling)
        x = F.relu(self.conv1(x))
        x = self.pool1(x) # Kích thước: /2
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x) # Kích thước: /4
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x) # Kích thước: /8 (16x16x16)
        
        # 2. Expansive Path (Upsampling - Khôi phục kích thước không gian)
        # Đây là điều chỉnh để phù hợp với yêu cầu Output 128x128x128
        
        x = F.relu(self.up_conv1(x)) # Kích thước: /4 (32x32x32)
        x = F.relu(self.up_conv2(x)) # Kích thước: /2 (64x64x64)
        
        # Thêm một lớp Conv để xử lý sau Up_conv3 để đạt 128x128x128
        x = F.relu(self.up_conv3(x)) # Kích thước: /1 (128x128x128)
        
        # Final output (Logits)
        logits = self.final_conv(x) # Kích thước: [B, num_classes, 128, 128, 128]
        
        # Không cần Softmax ở đây, sẽ được áp dụng sau hoặc trong hàm Loss
        return logits
    
    def predict_patch(self, patch):
        """ Dự đoán cho một patch (tensor PyTorch) """
        self.eval() # Chuyển sang chế độ đánh giá
        with torch.no_grad():
            output_logits = self(patch.unsqueeze(0).to(next(self.parameters()).device))
            # Áp dụng Softmax để có xác suất
            probabilities = F.softmax(output_logits.squeeze(0), dim=0) 
        self.train() # Trở lại chế độ huấn luyện
        return probabilities
    
    # Lưu ý: Các phương thức train() và predict_volume() nên được triển khai
    # trong một tập lệnh chính vì chúng liên quan đến DataLoader, Optimizer, và chu kỳ huấn luyện.