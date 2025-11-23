"""
Module Lựa Chọn Đặc Trưng VPT (VPT Feature Selection)
Sử dụng Vantage Point Tree để lựa chọn và tinh chọn đặc trưng quan trọng.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.spatial.distance import cdist, euclidean
import warnings


class VPTNode:
    """
    Node của Vantage Point Tree.
    
    Attributes:
        vantage_point: Vector điểm có lợi (vantage point)
        radius: Bán kính phân chia không gian
        left: Node con bên trái (distance <= radius)
        right: Node con bên phải (distance > radius)
        label: Nhãn của vantage point (nếu có)
    """
    
    def __init__(self, vantage_point, label=None, radius=0.0):
        self.vantage_point = vantage_point
        self.label = label
        self.radius = radius
        self.left = None
        self.right = None


class VPTSelector:
    """
    Class lựa chọn đặc trưng sử dụng Vantage Point Tree.
    
    Chức năng:
    - Xây dựng VPT từ training data
    - Tìm kiếm k-nearest neighbors hiệu quả
    - Lựa chọn N đặc trưng quan trọng nhất
    - Tính bản đồ xác suất tiên nghiệm
    
    Attributes:
        n_features (int): Số đặc trưng giữ lại (N)
        distance_metric (str): Metric khoảng cách ('euclidean', 'manhattan')
        selection_method (str): Phương pháp lựa chọn ('pca', 'kbest', 'variance')
        vpt_root: Root node của VPT tree
    """
    
    def __init__(self, n_features=10, distance_metric='euclidean', 
                 selection_method='variance', k_neighbors=5):
        """
        Khởi tạo VPTSelector.
        
        Args:
            n_features (int): Số đặc trưng giữ lại (N ≤ 20)
            distance_metric (str): 'euclidean' hoặc 'manhattan'
            selection_method (str): 'pca', 'kbest', 'variance', 'correlation'
            k_neighbors (int): Số láng giềng gần nhất để query
        """
        self.n_features = n_features
        self.distance_metric = distance_metric
        self.selection_method = selection_method
        self.k_neighbors = k_neighbors
        self.vpt_root = None
        self.feature_indices = None
        self.pca_model = None
        self.feature_importance = None
        
        print(f"[VPTSelector] Initialized:")
        print(f"  - Target features: {n_features}")
        print(f"  - Distance metric: {distance_metric}")
        print(f"  - Selection method: {selection_method}")
        print(f"  - K neighbors: {k_neighbors}")
    
    def _compute_distance(self, vec1, vec2):
        """
        Tính khoảng cách giữa 2 vectors.
        
        Args:
            vec1, vec2: Numpy arrays
            
        Returns:
            float: Khoảng cách
        """
        if self.distance_metric == 'euclidean':
            return euclidean(vec1, vec2)
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(vec1 - vec2))
        else:
            return euclidean(vec1, vec2)
    
    def _build_vpt_recursive(self, points, labels, depth=0, max_depth=20):
        """
        Xây dựng VPT tree đệ quy.
        
        Args:
            points (numpy.ndarray): Feature vectors (N, D)
            labels (numpy.ndarray): Labels (N,)
            depth (int): Độ sâu hiện tại
            max_depth (int): Độ sâu tối đa
            
        Returns:
            VPTNode: Root node của subtree
        """
        if len(points) == 0:
            return None
        
        if len(points) == 1 or depth >= max_depth:
            return VPTNode(points[0], label=labels[0] if labels is not None else None)
        
        # Chọn vantage point (random hoặc theo heuristic)
        vp_idx = np.random.randint(len(points))
        vantage_point = points[vp_idx]
        vp_label = labels[vp_idx] if labels is not None else None
        
        # Tính khoảng cách từ các điểm khác đến vantage point
        distances = np.array([self._compute_distance(vantage_point, p) for p in points])
        
        # Chọn median làm radius
        median_dist = np.median(distances)
        
        # Phân chia thành 2 groups (loại bỏ vantage point để không trùng lặp)
        left_mask = (distances <= median_dist) & (np.arange(len(points)) != vp_idx)
        right_mask = (distances > median_dist) & (np.arange(len(points)) != vp_idx)
        
        # Tạo node
        node = VPTNode(vantage_point, label=vp_label, radius=median_dist)
        
        # Xây dựng subtrees
        if np.any(left_mask):
            left_points = points[left_mask]
            left_labels = labels[left_mask] if labels is not None else None
            node.left = self._build_vpt_recursive(left_points, left_labels, depth + 1, max_depth)
        
        if np.any(right_mask):
            right_points = points[right_mask]
            right_labels = labels[right_mask] if labels is not None else None
            node.right = self._build_vpt_recursive(right_points, right_labels, depth + 1, max_depth)
        
        return node
    
    def build_tree(self, feature_vectors, labels=None):
        """
        Xây dựng VPT tree từ training data.
        
        Args:
            feature_vectors (numpy.ndarray): Feature vectors (N_samples, N_features)
            labels (numpy.ndarray): Labels cho supervised learning (optional)
            
        Returns:
            self
        """
        print(f"\n[VPTSelector] Building VPT tree...")
        print(f"  Training samples: {len(feature_vectors)}")
        print(f"  Feature dimension: {feature_vectors.shape[1]}")
        
        self.vpt_root = self._build_vpt_recursive(feature_vectors, labels)
        
        print(f"[VPTSelector] VPT tree built successfully!")
        return self
    
    def _query_nearest_recursive(self, node, query_point, k, results):
        """
        Tìm k nearest neighbors đệ quy trong VPT tree.
        
        Args:
            node: VPTNode hiện tại
            query_point: Vector cần query
            k: Số neighbors cần tìm
            results: List chứa (distance, node) được tìm thấy
        """
        if node is None:
            return
        
        # Tính khoảng cách đến vantage point
        dist = self._compute_distance(query_point, node.vantage_point)
        
        # Thêm vào results
        results.append((dist, node))
        results.sort(key=lambda x: x[0])
        if len(results) > k:
            results.pop()
        
        # Quyết định search subtree nào
        if len(results) < k or dist - node.radius < results[-1][0]:
            # Search left subtree
            self._query_nearest_recursive(node.left, query_point, k, results)
        
        if len(results) < k or dist + node.radius >= results[-1][0]:
            # Search right subtree
            self._query_nearest_recursive(node.right, query_point, k, results)
    
    def query_nearest(self, vector, k=None):
        """
        Tìm k láng giềng gần nhất cho vector.
        
        Args:
            vector (numpy.ndarray): Feature vector cần query
            k (int): Số neighbors (default: self.k_neighbors)
            
        Returns:
            list: List of (distance, VPTNode) tuples
        """
        if self.vpt_root is None:
            raise ValueError("VPT tree not built yet. Call build_tree() first.")
        
        k = k or self.k_neighbors
        results = []
        self._query_nearest_recursive(self.vpt_root, vector, k, results)
        return results
    
    def _select_features_variance(self, feature_tensor):
        """
        Lựa chọn features dựa trên variance (phương sai).
        
        Args:
            feature_tensor: (H, W, D, C) tensor
            
        Returns:
            numpy.ndarray: Indices của features được chọn
        """
        print(f"[VPTSelector] Selecting by variance...")
        
        # Tính variance cho mỗi channel
        num_channels = feature_tensor.shape[-1]
        variances = np.zeros(num_channels)
        
        for c in range(num_channels):
            variances[c] = feature_tensor[:, :, :, c].var()
        
        # Chọn top N channels có variance cao nhất
        top_indices = np.argsort(variances)[-self.n_features:][::-1]
        
        print(f"  Selected channels: {top_indices}")
        print(f"  Variance range: [{variances[top_indices].min():.4f}, {variances[top_indices].max():.4f}]")
        
        self.feature_importance = variances
        return top_indices
    
    def _select_features_correlation(self, feature_tensor):
        """
        Lựa chọn features dựa trên correlation (giảm redundancy).
        
        Args:
            feature_tensor: (H, W, D, C) tensor
            
        Returns:
            numpy.ndarray: Indices của features được chọn
        """
        print(f"[VPTSelector] Selecting by correlation (low redundancy)...")
        
        num_channels = feature_tensor.shape[-1]
        
        # Flatten spatial dimensions
        h, w, d, c = feature_tensor.shape
        reshaped = feature_tensor.reshape(-1, c)
        
        # Tính correlation matrix
        corr_matrix = np.corrcoef(reshaped.T)
        
        # Chọn features với correlation thấp (ít redundant)
        selected = []
        remaining = list(range(num_channels))
        
        # Chọn feature đầu tiên có variance cao nhất
        variances = [feature_tensor[:, :, :, i].var() for i in range(num_channels)]
        first = np.argmax(variances)
        selected.append(first)
        remaining.remove(first)
        
        # Chọn các features tiếp theo có correlation thấp với features đã chọn
        while len(selected) < self.n_features and remaining:
            min_corr = float('inf')
            best_idx = None
            
            for idx in remaining:
                # Tính avg correlation với features đã chọn
                avg_corr = np.mean([abs(corr_matrix[idx, s]) for s in selected])
                
                if avg_corr < min_corr:
                    min_corr = avg_corr
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        selected = np.array(selected)
        print(f"  Selected channels: {selected}")
        
        return selected
    
    def _select_features_pca(self, feature_tensor):
        """
        Lựa chọn features bằng PCA (Principal Component Analysis).
        
        Args:
            feature_tensor: (H, W, D, C) tensor
            
        Returns:
            numpy.ndarray: Transformed tensor với N components
        """
        print(f"[VPTSelector] Applying PCA...")
        
        h, w, d, c = feature_tensor.shape
        
        # Reshape để apply PCA
        reshaped = feature_tensor.reshape(-1, c)
        
        # Fit PCA
        self.pca_model = PCA(n_components=self.n_features)
        transformed = self.pca_model.fit_transform(reshaped)
        
        # Reshape back
        output = transformed.reshape(h, w, d, self.n_features)
        
        explained_var = self.pca_model.explained_variance_ratio_
        print(f"  Explained variance: {explained_var.sum():.4f}")
        print(f"  Components: {explained_var[:5]}")  # Show first 5
        
        return output
    
    def select_features(self, volume):
        """
        Lựa chọn N đặc trưng quan trọng nhất từ volume.
        
        Args:
            volume (numpy.ndarray): Feature tensor (H, W, D, C)
            
        Returns:
            numpy.ndarray: Selected feature tensor (H, W, D, N)
        """
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume (H, W, D, C), got shape {volume.shape}")
        
        h, w, d, num_channels = volume.shape
        
        print(f"\n{'='*60}")
        print(f"[VPTSelector] Starting feature selection...")
        print(f"{'='*60}")
        print(f"Input shape: {volume.shape}")
        print(f"Target: {num_channels} → {self.n_features} features")
        
        if self.n_features >= num_channels:
            print(f"[Warning] n_features ({self.n_features}) >= input channels ({num_channels})")
            print(f"[Warning] Returning original volume")
            return volume
        
        # Apply selection method
        if self.selection_method == 'pca':
            selected = self._select_features_pca(volume)
        elif self.selection_method == 'variance':
            indices = self._select_features_variance(volume)
            selected = volume[:, :, :, indices]
            self.feature_indices = indices
        elif self.selection_method == 'correlation':
            indices = self._select_features_correlation(volume)
            selected = volume[:, :, :, indices]
            self.feature_indices = indices
        else:
            # Default: variance
            indices = self._select_features_variance(volume)
            selected = volume[:, :, :, indices]
            self.feature_indices = indices
        
        print(f"\n[VPTSelector] Feature selection completed!")
        print(f"Output shape: {selected.shape}")
        print(f"{'='*60}\n")
        
        return selected
    
    def compute_prior_maps(self, volume, labeled_samples=None):
        """
        Tính bản đồ xác suất tiên nghiệm (prior probability maps).
        
        Sử dụng VPT để tìm neighbors và vote labels.
        
        Args:
            volume (numpy.ndarray): Feature tensor (H, W, D, C)
            labeled_samples (tuple): (feature_vectors, labels) từ training data
            
        Returns:
            numpy.ndarray: Prior probability maps (H, W, D)
        """
        if self.vpt_root is None and labeled_samples is not None:
            # Build tree nếu chưa có
            features, labels = labeled_samples
            self.build_tree(features, labels)
        
        if self.vpt_root is None:
            print("[Warning] No VPT tree available. Returning zeros.")
            return np.zeros(volume.shape[:3], dtype=np.float32)
        
        print(f"\n[VPTSelector] Computing prior probability maps...")
        
        h, w, d, c = volume.shape
        prior_maps = np.zeros((h, w, d), dtype=np.float32)
        
        # Sample sparse grid để tính prior (không tính toàn bộ voxels)
        stride = 5
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                for k in range(0, d, stride):
                    vector = volume[i, j, k, :]
                    
                    # Query nearest neighbors
                    neighbors = self.query_nearest(vector, k=self.k_neighbors)
                    
                    # Vote based on labels
                    if neighbors:
                        labels = [node.label for dist, node in neighbors if node.label is not None]
                        if labels:
                            # Probability = proportion of positive labels
                            prior_maps[i, j, k] = np.mean(labels)
        
        print(f"[VPTSelector] Prior maps computed!")
        print(f"  Mean probability: {prior_maps.mean():.4f}")
        print(f"  Std: {prior_maps.std():.4f}")
        
        return prior_maps
    
    def get_feature_importance(self):
        """
        Lấy feature importance scores.
        
        Returns:
            numpy.ndarray or None: Importance scores cho mỗi feature
        """
        return self.feature_importance
    
    def get_selected_indices(self):
        """
        Lấy indices của features đã chọn.
        
        Returns:
            numpy.ndarray or None: Indices của features
        """
        return self.feature_indices


if __name__ == "__main__":
    # Quick smoke test
    print("Running smoke test...")
    
    selector = VPTSelector(n_features=5, selection_method='variance')
    
    # Create synthetic feature volume (30×30×10×20)
    volume = np.random.randn(30, 30, 10, 20).astype(np.float32)
    print(f"\nSynthetic volume shape: {volume.shape}")
    
    # Test feature selection
    selected = selector.select_features(volume)
    print(f"Selected features shape: {selected.shape}")
    
    # Test VPT tree building
    training_samples = np.random.randn(100, 20).astype(np.float32)
    training_labels = np.random.randint(0, 2, size=100)
    
    selector.build_tree(training_samples, training_labels)
    
    # Test query
    query_vec = np.random.randn(20).astype(np.float32)
    neighbors = selector.query_nearest(query_vec, k=5)
    print(f"\nFound {len(neighbors)} neighbors")
    print(f"Distances: {[dist for dist, _ in neighbors]}")
