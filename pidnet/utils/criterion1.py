# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config

# ÖNEMLİ !!!!!!
# ÖNEMLİ !!!!!!
# config.MODEL.NUM_CLASSES == 1: Make sure that 
# config.MODEL.NUM_CLASSES = 1 is correctly set in your configuration YAML or Python file

import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class BinarySegmentationLoss(nn.Module):
    def __init__(self):
        super(BinarySegmentationLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, score, target):
        # BCE expects float targets in shape (N, 1, H, W)
        if target.dtype != torch.float:
            target = target.float()
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return self.criterion(score, target)


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')
    return loss


class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BoundaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        return bce_loss

    
if __name__ == '__main__':
    a = torch.zeros(2, 64, 64)
    a[:, 5, :] = 1  # Example: creating a simple target with a foreground pixel
    pre = torch.randn(2, 1, 64, 64)  # Example: random predicted outputs (logits)

    loss_fn = BinarySegmentationLoss()  # Use BinarySegmentationLoss for binary segmentation
    loss = loss_fn(pre, a)  # Compute the binary segmentation loss
    print("Binary segmentation loss:", loss)
"""