"""
Module 4: 3D CNN Model for Brain Tumor Segmentation
Implements a 3D Convolutional Neural Network for voxel classification
based on local context from VPT-selected features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class CNN3DNet(nn.Module):
    """PyTorch 3D CNN Network."""
    
    def __init__(self, in_channels: int = 4, num_classes: int = 4, input_shape: Tuple[int, int, int] = (128, 128, 128)):
        """
        Initialize the 3D CNN network.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            input_shape: Shape of input (H, W, D) - can be non-cubic
        """
        super(CNN3DNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Calculate flattened size after conv layers
        # After 3 pooling layers (each /2): input_shape // 8 for each dimension
        H, W, D = input_shape
        feat_h = H // 8
        feat_w = W // 8
        feat_d = D // 8
        self.flatten_size = 128 * feat_h * feat_w * feat_d
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """Forward pass."""
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.reshape(x.size(0), -1)  # Flatten (use reshape instead of view for compatibility)
        
        x = self.dropout1(self.relu4(self.fc1(x)))
        x = self.dropout2(self.relu5(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class CNN3DModel:
    """
    3D Convolutional Neural Network for brain tumor segmentation.
    
    Architecture:
    - Multiple 3D convolutional layers with increasing filters (32 -> 64 -> 128)
    - Max pooling layers for spatial dimension reduction
    - Fully connected layers for classification
    - Softmax activation for multi-class segmentation
    
    Classes:
    - 0: Background
    - 1: Necrotic core
    - 2: Edema
    - 4: Enhancing tumor (mapped to index 3 internally)
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (128, 128, 128, 4), 
                 num_classes: int = 4):
        """
        Initialize the 3D CNN model.
        
        Args:
            input_shape: Shape of input patches (height, width, depth, channels)
                        Default: (128, 128, 128, 4) for 128x128x128 patches with 4 features
            num_classes: Number of segmentation classes (default: 4)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = None
        
        # Build the model architecture
        self.build_model()
    
    def build_model(self) -> None:
        """
        Build the 3D CNN architecture with conv, pooling, and dense layers.
        
        Architecture details:
        - Conv3D layer 1: 32 filters, 3x3x3 kernel, ReLU activation
        - MaxPooling3D: 2x2x2 pool size
        - Conv3D layer 2: 64 filters, 3x3x3 kernel, ReLU activation
        - MaxPooling3D: 2x2x2 pool size
        - Conv3D layer 3: 128 filters, 3x3x3 kernel, ReLU activation
        - MaxPooling3D: 2x2x2 pool size
        - Flatten layer
        - Dense layer: 256 units, ReLU activation
        - Dropout: 0.5 rate
        - Dense layer: 128 units, ReLU activation
        - Dropout: 0.3 rate
        - Output layer: num_classes units, Softmax activation
        """
        in_channels = self.input_shape[3]  # Last dimension is channels
        input_spatial = self.input_shape[:3]  # (H, W, D)
        self.model = CNN3DNet(in_channels=in_channels, num_classes=self.num_classes, 
                             input_shape=input_spatial)
        self.model = self.model.to(self.device)
    
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
        Train the 3D CNN model.
        
        Args:
            train_data: Tuple of (X_train, y_train)
                       X_train shape: (n_samples, 128, 128, 128, n_features)
                       y_train shape: (n_samples, num_classes) - one-hot encoded
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
        y_train = torch.FloatTensor(y_train)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            X_val, y_val = val_data
            X_val = torch.FloatTensor(X_val).permute(0, 4, 1, 2, 3)
            y_val = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5, min_lr=1e-7)
        
        # Training history
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                # Convert one-hot to class indices
                _, labels = torch.max(batch_y, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            history['loss'].append(avg_train_loss)
            history['accuracy'].append(train_accuracy)
            
            # Validation phase
            if val_data:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        _, labels = torch.max(batch_y, 1)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct / val_total
                history['val_loss'].append(avg_val_loss)
                history['val_accuracy'].append(val_accuracy)
                
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
                      f"acc: {train_accuracy:.4f}, val_loss: {avg_val_loss:.4f}, "
                      f"val_acc: {val_accuracy:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}, "
                      f"acc: {train_accuracy:.4f}")
        
        self.history = history
        return history
    
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Predict segmentation for a single patch.
        
        Args:
            patch: Input patch of shape (128, 128, 128, n_features) or (1, 128, 128, 128, n_features)
            
        Returns:
            Probability distribution of shape (num_classes,)
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
        
        # Return probabilities for the single sample
        return probabilities[0].cpu().numpy()
    
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
                    
                    # Predict - returns (num_classes,)
                    pred = self.predict_patch(patch)
                    
                    # Broadcast prediction to entire patch
                    # Each voxel in patch gets the same class probabilities
                    for i in range(pH):
                        for j in range(pW):
                            for k in range(pD):
                                output[h+i, w+j, d+k, :] += pred
                                count[h+i, w+j, d+k, 0] += 1
        
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
        else:
            print("Model not built yet.")
