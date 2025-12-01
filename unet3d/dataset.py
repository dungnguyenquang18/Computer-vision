import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

class BratsDataset(Dataset):
    def __init__(self, root_dir, patch_size=128):
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(root_dir))
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def load_nii(self, path):
        return nib.load(path).get_fdata().astype(np.float32)

    def random_crop(self, img, mask, size):
        _, D, H, W = img.shape

        # đảm bảo D/H/W > size
        d = np.random.randint(0, max(1, D - size))
        h = np.random.randint(0, max(1, H - size))
        w = np.random.randint(0, max(1, W - size))

        return (
            img[:, d:d+size, h:h+size, w:w+size],
            mask[d:d+size, h:h+size, w:w+size]
        )

    def __getitem__(self, idx):
        case = self.samples[idx]
        folder = os.path.join(self.root_dir, case)

        # flair = self.load_nii(os.path.join(folder, case + "_flair.nii.gz"))
        t1 = self.load_nii(os.path.join(folder, case + "_t1.nii.gz"))
        # t1ce = self.load_nii(os.path.join(folder, case + "_t1ce.nii.gz"))
        # t2 = self.load_nii(os.path.join(folder, case + "_t2.nii.gz"))
        mask = self.load_nii(os.path.join(folder, case + "_seg.nii.gz"))
        mask[mask == 4] = 3  # convert ET label 4 → 3

        # stack modal
        image = np.stack([ t1], axis=0)
        # image = np.stack([flair, t1, t1ce, t2], axis=0)
        # normalize
        image = (image - image.mean()) / (image.std() + 1e-6)

        # crop patch 128³
        image, mask = self.random_crop(image, mask, self.patch_size)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long)
        )
