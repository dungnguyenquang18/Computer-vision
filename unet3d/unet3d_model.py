"""
Module 5: 3D U-Net Model for Brain Tumor Segmentation
Implements a 3D U-Net architecture specialized for semantic segmentation
with skip connections to preserve spatial information.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class DoubleConv3D(nn.Module):
    """Double convolution block: Conv3D -> ReLU -> Conv3D -> ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel (default: 3)
        """
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: DoubleConv -> MaxPool"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block: UpConv -> Concatenate with skip -> DoubleConv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize decoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)  # in_channels because of concatenation
    
    def forward(self, x, skip):
        x = self.upconv(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet3DNet(nn.Module):
    """PyTorch 3D U-Net Network with encoder-decoder architecture and skip connections."""
    
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base_filters: int = 32):
        """
        Initialize the 3D U-Net network.
        
        Architecture:
        - Encoder: 4 levels with increasing filters (32 -> 64 -> 128 -> 256)
        - Bottleneck: Double conv with 256 filters
        - Decoder: 4 levels with decreasing filters (256 -> 128 -> 64 -> 32)
        - Skip connections between encoder and decoder at each level
        
        Args:
            in_channels: Number of input channels (default: 4 for 4 MRI modalities)
            num_classes: Number of output classes (default: 4)
            base_filters: Number of filters in first layer (default: 32)
        """
        super(UNet3DNet, self).__init__()
        
        # Encoder (contracting path)
        self.encoder1 = EncoderBlock(in_channels, base_filters)
        self.encoder2 = EncoderBlock(base_filters, base_filters * 2)
        self.encoder3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.encoder4 = EncoderBlock(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(base_filters * 8, base_filters * 16)
        
        # Decoder (expanding path)
        self.decoder4 = DecoderBlock(base_filters * 16, base_filters * 8)
        self.decoder3 = DecoderBlock(base_filters * 8, base_filters * 4)
        self.decoder2 = DecoderBlock(base_filters * 4, base_filters * 2)
        self.decoder1 = DecoderBlock(base_filters * 2, base_filters)
        
        # Output layer: 1x1x1 convolution
        self.output = nn.Conv3d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (N, C, H, W, D)
            
        Returns:
            Output tensor of shape (N, num_classes, H, W, D)
        """
        # Encoder path with skip connections
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        
        # Output layer
        x = self.output(x)
        
        return x


class UNet3DModel:
    """
    3D U-Net for brain tumor segmentation with skip connections.
    
    Architecture:
    - Contracting path (encoder): extracts context and reduces spatial dimensions
    - Expanding path (decoder): up-samples and recovers spatial information
    - Skip connections: concatenate feature maps from encoder to decoder
    - Preserves fine-grained spatial details for accurate boundary segmentation
    
    Classes:
    - 0: Background
    - 1: Necrotic core
    - 2: Edema
    - 4: Enhancing tumor (mapped to index 3 internally)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (128, 128, 128, 4), 
                 num_classes: int = 4, base_filters: int = 32):
        """
        Initialize the 3D U-Net model.
        
        Args:
            input_shape: Shape of input patches (height, width, depth, channels)
                        Default: (128, 128, 128, 4) for 128x128x128 patches with 4 features
            num_classes: Number of segmentation classes (default: 4)
            base_filters: Number of filters in first layer (default: 32)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = None
        
        # Build the model architecture
        self.build_model()
    
    def build_model(self) -> None:
        """
        Build the 3D U-Net architecture.
        
        Architecture details:
        Encoder (Contracting Path):
        - Level 1: Conv3D(32) -> ReLU -> Conv3D(32) -> ReLU -> MaxPool
        - Level 2: Conv3D(64) -> ReLU -> Conv3D(64) -> ReLU -> MaxPool
        - Level 3: Conv3D(128) -> ReLU -> Conv3D(128) -> ReLU -> MaxPool
        - Level 4: Conv3D(256) -> ReLU -> Conv3D(256) -> ReLU -> MaxPool
        
        Bottleneck:
        - Conv3D(512) -> ReLU -> Conv3D(512) -> ReLU
        
        Decoder (Expanding Path):
        - Level 4: UpConv(256) -> Concatenate skip4 -> DoubleConv(256)
        - Level 3: UpConv(128) -> Concatenate skip3 -> DoubleConv(128)
        - Level 2: UpConv(64) -> Concatenate skip2 -> DoubleConv(64)
        - Level 1: UpConv(32) -> Concatenate skip1 -> DoubleConv(32)
        
        Output Layer:
        - Conv3D(num_classes) with 1x1x1 kernel
        """
        in_channels = self.input_shape[3]  # Last dimension is channels
        self.model = UNet3DNet(in_channels=in_channels, num_classes=self.num_classes,
                              base_filters=self.base_filters)
        self.model = self.model.to(self.device)
    
    def build_encoder(self) -> List[nn.Module]:
        """
        Build the contracting path (encoder).
        
        Returns:
            List of encoder blocks
        """
        # This method exists for API compatibility
        # The encoder is already built in build_model()
        encoder_blocks = [
            self.model.encoder1,
            self.model.encoder2,
            self.model.encoder3,
            self.model.encoder4
        ]
        return encoder_blocks
    
    def build_decoder(self) -> List[nn.Module]:
        """
        Build the expanding path (decoder).
        
        Returns:
            List of decoder blocks
        """
        # This method exists for API compatibility
        # The decoder is already built in build_model()
        decoder_blocks = [
            self.model.decoder4,
            self.model.decoder3,
            self.model.decoder2,
            self.model.decoder1
        ]
        return decoder_blocks
    
    @staticmethod
    def _dice_loss(pred, target, smooth=1e-6):
        """
        Calculate Dice loss for segmentation.
        
        Args:
            pred: Predicted probabilities (after softmax)
            target: Ground truth labels (one-hot encoded)
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice loss value
        """
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1e-6):
        """
        Calculate Dice coefficient metric for segmentation evaluation.
        
        Args:
            y_true: Ground truth labels (torch.Tensor)
            y_pred: Predicted labels (torch.Tensor)
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient score
        """
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = (y_true_f * y_pred_f).sum()
        return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              epochs: int = 100, batch_size: int = 2,
              save_path: Optional[str] = None) -> dict:
        """
        Train the 3D U-Net model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
                       X_train shape: (n_samples, 128, 128, 128, n_features)
                       y_train shape: (n_samples, 128, 128, 128) - segmentation masks
            val_data: Optional tuple of (X_val, y_val) for validation
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size for training (default: 2)
            save_path: Optional path to save the best model
            
        Returns:
            Training history dictionary
        """
        X_train, y_train = train_data
        
        # Convert to PyTorch tensors and change format from (N,H,W,D,C) to (N,C,H,W,D)
        X_train = torch.FloatTensor(X_train).permute(0, 4, 1, 2, 3)
        y_train = torch.LongTensor(y_train)  # Segmentation masks are LongTensor
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            X_val, y_val = val_data
            X_val = torch.FloatTensor(X_val).permute(0, 4, 1, 2, 3)
            y_val = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, min_lr=1e-7)
        
        # Training history
        history = {'loss': [], 'dice': [], 'val_loss': [], 'val_dice': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_dice = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Calculate loss
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate Dice coefficient
                with torch.no_grad():
                    pred_probs = torch.softmax(outputs, dim=1)
                    pred_labels = torch.argmax(pred_probs, dim=1)
                    dice = self._dice_coefficient(batch_y, pred_labels)
                    train_dice += dice.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_dice = train_dice / len(train_loader)
            history['loss'].append(avg_train_loss)
            history['dice'].append(avg_train_dice)
            
            # Validation phase
            if val_data:
                self.model.eval()
                val_loss = 0.0
                val_dice = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        
                        # Calculate Dice coefficient
                        pred_probs = torch.softmax(outputs, dim=1)
                        pred_labels = torch.argmax(pred_probs, dim=1)
                        dice = self._dice_coefficient(batch_y, pred_labels)
                        val_dice += dice.item()
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_dice = val_dice / len(val_loader)
                history['val_loss'].append(avg_val_loss)
                history['val_dice'].append(avg_val_dice)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}, "
                      f"dice: {avg_train_dice:.4f}, val_loss: {avg_val_loss:.4f}, "
                      f"val_dice: {avg_val_dice:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}, "
                      f"dice: {avg_train_dice:.4f}")
        
        self.history = history
        return history
    
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Predict segmentation for a single patch.
        
        Args:
            patch: Input patch of shape (128, 128, 128, n_features) or (1, 128, 128, 128, n_features)
            
        Returns:
            Probability map of shape (128, 128, 128, num_classes)
        """
        self.model.eval()
        
        if len(patch.shape) == 4:
            # Add batch dimension if not present
            patch = np.expand_dims(patch, axis=0)
        
        # Convert to tensor and change format from (N,H,W,D,C) to (N,C,H,W,D)
        patch_tensor = torch.FloatTensor(patch).permute(0, 4, 1, 2, 3).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(patch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # Convert back to (N,H,W,D,C) format
        probabilities = probabilities.permute(0, 2, 3, 4, 1).cpu().numpy()
        
        # Return probability map for the single sample
        return probabilities[0]
    
    def predict_volume(self, volume: np.ndarray, 
                      patch_size: Tuple[int, int, int] = (128, 128, 128),
                      stride: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Predict segmentation for a full volume by processing patches.
        
        Args:
            volume: Input volume of shape (H, W, D, n_features)
                   e.g., (240, 240, 160, n_features)
            patch_size: Size of patches to extract (default: 128x128x128)
            stride: Stride for patch extraction. If None, uses patch_size (no overlap)
            
        Returns:
            Probability map of shape (H, W, D, num_classes)
        """
        self.model.eval()
        
        if stride is None:
            stride = patch_size
        
        H, W, D, C = volume.shape
        pH, pW, pD = patch_size
        sH, sW, sD = stride
        
        # Initialize output volume
        output = np.zeros((H, W, D, self.num_classes), dtype=np.float32)
        count = np.zeros((H, W, D, 1), dtype=np.float32)
        
        # Extract and predict patches
        for h in range(0, H - pH + 1, sH):
            for w in range(0, W - pW + 1, sW):
                for d in range(0, D - pD + 1, sD):
                    # Extract patch
                    patch = volume[h:h+pH, w:w+pW, d:d+pD, :]
                    
                    # Predict - returns (pH, pW, pD, num_classes)
                    pred = self.predict_patch(patch)
                    
                    # Accumulate predictions
                    output[h:h+pH, w:w+pW, d:d+pD, :] += pred
                    count[h:h+pH, w:w+pW, d:d+pD, 0] += 1
        
        # Average overlapping predictions
        output = np.divide(output, count, 
                          out=np.zeros_like(output), 
                          where=count != 0)
        
        return output
    
    def get_segmentation_mask(self, probability_map: np.ndarray) -> np.ndarray:
        """
        Convert probability map to segmentation mask.
        
        Args:
            probability_map: Probability map of shape (H, W, D, num_classes)
            
        Returns:
            Segmentation mask of shape (H, W, D) with class labels
            Note: Class 3 is mapped to label 4 (enhancing tumor)
        """
        # Get class with highest probability
        mask = np.argmax(probability_map, axis=-1)
        
        # Map class 3 to label 4 (enhancing tumor)
        mask = np.where(mask == 3, 4, mask)
        
        return mask.astype(np.uint8)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'base_filters': self.base_filters,
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', None)
        print(f"Model loaded from {filepath}")
    
    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model:
            print(self.model)
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"\nArchitecture: U-Net with skip connections")
            print(f"Input shape: {self.input_shape}")
            print(f"Output classes: {self.num_classes}")
            print(f"Base filters: {self.base_filters}")
        else:
            print("Model not built yet.")
