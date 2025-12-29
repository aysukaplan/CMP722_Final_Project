import os
import numpy as np
from PIL import Image
from .base_dataset import BaseDataset  # Adjust this import if your BaseDataset is elsewhere

class CustomRoadSegmentationDataset(BaseDataset):
    def __init__(self, root, split='train', **kwargs):
        super(CustomRoadSegmentationDataset, self).__init__(**kwargs)

        self.split = split
        self.root = root

        self.image_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'masks', split)

        self.image_paths = sorted([
            os.path.join(self.image_dir, fname)
            for fname in os.listdir(self.image_dir)
            if fname.endswith('.jpg') or fname.endswith('.png')
        ])

        self.mask_paths = sorted([
            os.path.join(self.mask_dir, fname)
            for fname in os.listdir(self.mask_dir)
            if fname.endswith('.png')
        ])

        assert len(self.image_paths) == len(self.mask_paths), \
            f"#images ({len(self.image_paths)}) != #masks ({len(self.mask_paths)})"

        self.files = list(zip(self.image_paths, self.mask_paths))

    def __getitem__(self, index):
        image_path, mask_path = self.files[index]

        # Load image and mask
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))

        # Convert mask to binary (0 for background, 1 for road)
        mask = (mask > 0).astype(np.uint8)
        
        image, label, edge = self.gen_sample(image, mask)

        return image.copy(), label.copy(), edge.copy(), os.path.basename(image_path)
    



