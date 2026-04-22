from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(128, 96)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize(self.img_size, Image.NEAREST)
        img = np.array(img).astype(np.float32) / 255.0

        mask = Image.open(self.mask_paths[idx]).convert("RGB")
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return img, mask