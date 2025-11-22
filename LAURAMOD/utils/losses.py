import torch
import torch.nn as nn
from monai.losses import DiceCELoss
import logging

logger = logging.getLogger(__name__)

class StableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)

    def forward(self, seg_preds, seg_labels):
        if torch.isnan(seg_preds).any():
            logger.warning("NaN detected in predictions! Returning large finite loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        if torch.isnan(seg_labels).any():
            raise ValueError("NaN detected in ground truth labels")
        
        seg_preds = torch.clamp(seg_preds, min=-10, max=10)
        loss = self.seg_loss(seg_preds, seg_labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN or Inf loss detected! Returning fallback loss.")
            return torch.tensor(1000.0, requires_grad=True, device=seg_preds.device)
        
        return loss
