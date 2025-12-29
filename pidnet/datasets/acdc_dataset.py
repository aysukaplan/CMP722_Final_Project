# acdc_dataloader.py
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import numpy as np
"""
# Cityscapes mean & std
mean = [0.286, 0.325, 0.283]
std  = [0.176, 0.180, 0.177]

"""
# Imagenet mean & std
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def input_transform(image):
    image = image.astype(np.float32) / 255.0
    image -= mean
    image /= std
    return image

class ACDCSubDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.images = sorted(list(self.image_dir.rglob("*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        image = input_transform(image)  # normalize etc.

        # mask path
        relative_path = img_path.relative_to(self.image_dir)
        mask_name = relative_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        mask_path = self.mask_dir / relative_path.parent / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask = np.array(Image.open(mask_path)) # shape (H, W, 3)

        # Convert to torch tensor
        image = torch.from_numpy(image.transpose(2,0,1)).float()
        mask = torch.from_numpy(mask).long()


        return image, mask


def get_acdc_val_loaders(base_image_dir, base_mask_dir, batch_size=1, num_workers=2, shuffle=False):
    subdatasets = ["fog", "night", "rain", "snow"]
    loaders = {}
    for name in subdatasets:
        image_dir = f"{base_image_dir}/{name}/val"
        mask_dir  = f"{base_mask_dir}/{name}/val"
        dataset = ACDCSubDataset(image_dir=image_dir, mask_dir=mask_dir)
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loaders
def get_acdc_train_loaders(base_image_dir, base_mask_dir, batch_size=1, num_workers=2, shuffle=True):
    subdatasets = ["fog", "night", "rain", "snow"]
    loaders = {}
    
    for name in subdatasets:
        image_dir = f"{base_image_dir}/{name}/train"
        mask_dir  = f"{base_mask_dir}/{name}/train"
        dataset = ACDCSubDataset(image_dir=image_dir, mask_dir=mask_dir)
        loaders[name] = DataLoader(dataset, 
                                   batch_size=batch_size, 
                                   shuffle=shuffle, 
                                   num_workers=num_workers)
    return loaders
