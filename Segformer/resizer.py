# Save this file as resizer.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

class Resizer(nn.Module):
    def __init__(self, scale_factor):
        super(Resizer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # 1. Get current dimensions (Height, Width)
        h, w = x.shape[-2:]
        
        # 2. Calculate new dimensions explicitly
        #    (We use int() to ensure they are valid pixel dimensions)
        new_h = int(h * self.scale_factor)
        new_w = int(w * self.scale_factor)
        
        # 3. Resize using strict 'size' argument (Safest method)
        return F.interpolate(x, 
                             size=(new_h, new_w), 
                             mode='bilinear', 
                             align_corners=False)