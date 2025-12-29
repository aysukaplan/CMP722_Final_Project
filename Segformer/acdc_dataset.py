from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random

"""
# Cityscapes mean & std
mean = [0.286, 0.325, 0.283]
std = [0.176, 0.180, 0.177]
"""
# Imagenet mean & std
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def input_transform(image):
    # Input for VAE
    image = image.astype(np.float32) / 255.0
    return image

class ACDCSubDataset(Dataset):
    def __init__(self, image_dir, mask_dir, 
                 patch_size=(1024, 1024), 
                 image_size=(1080, 1920), 
                 stride=(512, 512),
                 max_images=None, 
                 seed=42):         
        
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        
        # 1. Sort first to ensure consistent starting list
        all_images = sorted(list(self.image_dir.rglob("*.png")))
        
        # 2. Deterministic Sampling using the specific seed
        if max_images is not None and max_images < len(all_images):
            # Create a local RNG specifically for this selection
            rng = random.Random(seed) 
            rng.shuffle(all_images)
            self.images = all_images[:max_images]
            # Sort again for predictable processing order (alphanumeric)
            self.images = sorted(self.images)
        else:
            self.images = all_images

        # Patch params
        self.img_h, self.img_w = image_size
        self.patch_h, self.patch_w = patch_size
        self.stride_h, self.stride_w = stride

        if self.img_h <= self.patch_h: self.stride_h = self.patch_h
        if self.img_w <= self.patch_w: self.stride_w = self.patch_w

        self.n_patches_h = int(np.ceil((self.img_h - self.patch_h) / self.stride_h)) + 1
        self.n_patches_w = int(np.ceil((self.img_w - self.patch_w) / self.stride_w)) + 1
        
        if self.img_h <= self.patch_h: self.n_patches_h = 1
        if self.img_w <= self.patch_w: self.n_patches_w = 1

        self.patches_per_image = self.n_patches_h * self.n_patches_w
        self.total_patches = len(self.images) * self.patches_per_image

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        # Index Mapping
        image_index = idx // self.patches_per_image
        patch_index_in_image = idx % self.patches_per_image
        patch_y_index = patch_index_in_image // self.n_patches_w
        patch_x_index = patch_index_in_image % self.n_patches_w
 
        # Coordinates
        y_start = patch_y_index * self.stride_h
        x_start = patch_x_index * self.stride_w

        if y_start + self.patch_h > self.img_h:
            y_start = self.img_h - self.patch_h
        if x_start + self.patch_w > self.img_w:
            x_start = self.img_w - self.patch_w
        
        y_start = max(0, y_start)
        x_start = max(0, x_start)

        # Load
        img_path = self.images[image_index]
        full_image = np.array(Image.open(img_path).convert("RGB"))

        relative_path = img_path.relative_to(self.image_dir)
        mask_name = relative_path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        mask_path = self.mask_dir / relative_path.parent / mask_name
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        full_mask = np.array(Image.open(mask_path)) 

        # Crop
        image_patch = full_image[y_start : y_start + self.patch_h, x_start : x_start + self.patch_w, :]
        mask_patch = full_mask[y_start : y_start + self.patch_h, x_start : x_start + self.patch_w]

        # Transform
        image_patch_transformed = input_transform(image_patch)
        image_patch_tensor = torch.from_numpy(image_patch_transformed.transpose(2,0,1)).float()
        mask_patch_tensor = torch.from_numpy(mask_patch.copy()).long()

        coords = (image_index, y_start, x_start)

        return image_patch_tensor, mask_patch_tensor, coords, img_path.name

# -----------------------------------------------------------------------------
#  Loader Functions with Worker Seeding Closure
# -----------------------------------------------------------------------------

def get_acdc_val_loaders(base_image_dir, base_mask_dir, batch_size=1, num_workers=1, 
                         patch_size=(1024, 1024), image_size=(1080, 1920), stride=(512, 512),
                         max_images=10, seed=42): 
    
    # --- DEFINE WORKER INIT INSIDE TO CAPTURE 'seed' ---
    def seed_worker(worker_id):
        """Seeds each worker's RNG state based on the passed 'seed' argument."""
        # We add worker_id to the base seed so every worker gets a unique deterministic seed
        worker_seed = seed + worker_id 
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # ---------------------------------------------------

    subdatasets = ["fog", "night", "rain", "snow"]
    loaders = {}
    shuffle = False 
    
    for name in subdatasets:
        image_dir = f"{base_image_dir}/{name}/val"
        mask_dir = f"{base_mask_dir}/{name}/val"
        
        dataset = ACDCSubDataset(
            image_dir=image_dir, 
            mask_dir=mask_dir,
            patch_size=patch_size, 
            image_size=image_size, 
            stride=stride,
            max_images=max_images, 
            seed=seed
        )
        
        loaders[name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            worker_init_fn=seed_worker # <--- Pass the closure
        )
    
    return loaders

def get_acdc_train_loaders(base_image_dir, base_mask_dir, batch_size=1, num_workers=1, 
                           patch_size=(1024, 1024), image_size=(1080, 1920), stride=(512, 512),
                           max_images=None, seed=42):
    
    # --- DEFINE WORKER INIT INSIDE TO CAPTURE 'seed' ---
    def seed_worker(worker_id):
        worker_seed = seed + worker_id 
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    # ---------------------------------------------------

    subdatasets = ["fog", "night", "rain", "snow"]
    loaders = {}
    shuffle = False 
    
    for name in subdatasets:
        image_dir = f"{base_image_dir}/{name}/train"
        mask_dir = f"{base_mask_dir}/{name}/train"
        
        dataset = ACDCSubDataset(
            image_dir=image_dir, 
            mask_dir=mask_dir,
            patch_size=patch_size, 
            image_size=image_size, 
            stride=stride,
            max_images=max_images,
            seed=seed
        )
        
        loaders[name] = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers,
            worker_init_fn=seed_worker
        )
    return loaders