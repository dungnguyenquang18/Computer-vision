# Brain Tumor Segmentation Pipeline

Automated brain tumor segmentation using GLCM features, VPT selection, 3D CNN, and 3D U-Net.

## Overview

This pipeline segments brain tumors from multimodal MRI scans (T1, T1ce, T2, FLAIR) into:

- **Label 0**: Background
- **Label 1**: Necrotic Core
- **Label 2**: Edema
- **Label 4**: Enhancing Tumor

## Pipeline Architecture

```
MRI Input (240×240×155×4)
    ↓
1. Preprocessing (padding + normalization)
    ↓
2. GLCM Feature Extraction (20 features)
    ↓
3. VPT Feature Selection (top N features)
    ↓
4. 3D CNN Prediction
    ↓
5. 3D U-Net Prediction
    ↓
6. Ensemble (weighted average)
    ↓
Segmentation Mask (240×240×155)
```

## Installation

```bash
# Required packages
pip install torch numpy scipy scikit-learn scikit-image nibabel

# Optional for GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Training

Train CNN and U-Net models on your data:

```bash
# Train U-Net (recommended)
python train.py --model unet --data ../BraTS_data --epochs 50 --batch 2

# Train CNN
python train.py --model cnn --data ../BraTS_data --epochs 50 --batch 2
```

**Arguments:**

- `--model`: Model type (`cnn` or `unet`)
- `--data`: Path to BraTS data directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 2)
- `--save`: Directory to save models (default: `./models`)

**Expected Data Structure:**

```
BraTS_data/
├── BraTS2021_00001_t1.nii.gz
├── BraTS2021_00001_t1ce.nii.gz
├── BraTS2021_00001_t2.nii.gz
├── BraTS2021_00001_flair.nii.gz
├── BraTS2021_00001_seg.nii.gz
└── ...
```

### 2. Inference

Run segmentation on new patient data:

```bash
# With trained models
python inference.py --patient BraTS2021_00451 --data ../BraTS_data --models ./models --trained

# Without trained models (testing pipeline only)
python inference.py --patient BraTS2021_00451 --data ../BraTS_data

# Fast mode (uses stride for GLCM)
python inference.py --patient BraTS2021_00451 --data ../BraTS_data --trained --fast
```

**Arguments:**

- `--patient`: Patient ID
- `--data`: Path to input data directory
- `--models`: Path to trained models directory (default: `./models`)
- `--output`: Path to output directory (default: `./predictions`)
- `--trained`: Use trained models (default: False)
- `--fast`: Use fast mode (default: False)

**Output:**

- `predictions/BraTS2021_00451_prediction.npy`: Numpy array
- `predictions/BraTS2021_00451_prediction.nii.gz`: NIfTI format

### 3. Testing

Evaluate trained models on test set:

```bash
python test_trained.py --patients BraTS2021_00451 BraTS2021_00452 --data ../BraTS_data --models ./models
```

**Arguments:**

- `--patients`: List of patient IDs (space-separated)
- `--data`: Path to test data directory
- `--models`: Path to trained models directory
- `--output`: Output file for results (default: `test_results.txt`)

**Output:**

- Individual Dice scores for each patient
- Aggregate statistics (mean ± std)
- Saved to `test_results.txt`

## Quick Tests

### Test with Synthetic Data

```bash
python test_simple.py
```

### Test with Real Data (Cropped)

```bash
python test_quick.py
```

## Module Structure

```
Computer-vision/
├── preprocessing/          # Preprocessing (padding, normalization)
├── glcm_extraction/        # GLCM feature extraction
├── vpt_selection/          # VPT feature selection
├── cnn3d/                  # 3D CNN model
├── unet3d/                 # 3D U-Net model
├── ensemble/               # Ensemble model
├── full_pipeline.py        # Complete pipeline
├── train.py                # Training script
├── inference.py            # Inference script
├── test_trained.py         # Testing script
└── README.md               # This file
```

## API Usage

### Full Pipeline

```python
from full_pipeline import BrainTumorSegmentationPipeline
import numpy as np

# Initialize pipeline
pipeline = BrainTumorSegmentationPipeline(
    target_shape=(240, 240, 160, 4),
    glcm_window_size=5,
    glcm_levels=16,
    vpt_n_features=10,
    patch_size=(128, 128, 128),
    num_classes=4,
    ensemble_strategy='weighted'
)

# Load your MRI data (240, 240, 155, 4)
volume = np.random.randn(240, 240, 155, 4).astype(np.float32)

# Run segmentation
mask = pipeline.predict(volume, fast_mode=False)

# mask shape: (240, 240, 155)
```

### Individual Models

```python
from cnn3d_model import CNN3DModel
from unet3d_model import UNet3DModel

# CNN model
cnn = CNN3DModel(input_shape=(128, 128, 128, 10), num_classes=4)
cnn.load_model('./models/cnn_best.pth')

# U-Net model
unet = UNet3DModel(input_shape=(128, 128, 128, 10), num_classes=4)
unet.load_model('./models/unet_best.pth')

# Predict
prob_cnn = cnn.predict_patch(patch)
prob_unet = unet.predict_patch(patch)
```

## Performance Tips

1. **GPU**: Use CUDA for 10-20x speedup

   ```python
   pipeline = BrainTumorSegmentationPipeline(device='cuda')
   ```

2. **Fast Mode**: Use for quick testing (reduces accuracy)

   ```python
   mask = pipeline.predict(volume, fast_mode=True, glcm_stride=10)
   ```

3. **Batch Size**: Adjust based on GPU memory

   ```bash
   python train.py --batch 4  # If you have more GPU memory
   ```

4. **Patch Size**: Smaller patches = less memory, more patches
   ```python
   pipeline = BrainTumorSegmentationPipeline(patch_size=(64, 64, 64))
   ```

## Expected Results

**Untrained Models** (random predictions):

- Dice scores: ~0.01-0.10 (very low)
- For pipeline testing only

**Trained Models** (after proper training):

- Necrotic Core: 0.70-0.85
- Edema: 0.75-0.88
- Enhancing Tumor: 0.75-0.85
- Mean Dice: 0.73-0.86

_Note: Actual results depend on training data quality and quantity_

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `patch_size`
- Use `fast_mode=True`
- Use CPU instead of GPU

### Slow Training

- Use GPU (`device='cuda'`)
- Enable `fast_mode` for GLCM
- Reduce number of training samples

### Poor Segmentation

- Train models on more data
- Increase training epochs
- Adjust ensemble weights
- Use larger patch size

## Citation

If you use this pipeline, please cite:

```
@article{brats2021,
  title={Brain Tumor Segmentation with GLCM-VPT-DL Pipeline},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
