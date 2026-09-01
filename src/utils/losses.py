"""
src/utils/losses.py
===================
Compound 3D Multi-Class Dice + Cross-Entropy Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss3D(nn.Module):
    """
    Compound 3D Multi-Class Dice + Cross-Entropy Loss.
    """
    def __init__(
        self,
        num_classes: int = 14,
        lambda_dice: float = 0.5,
        smooth: float = 1e-5,
        ignore_background: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).contiguous().float()

        start_c = 1 if self.ignore_background else 0
        dice_scores = []
        for c in range(start_c, self.num_classes):
            p_c = probs[:, c].reshape(probs.shape[0], -1)
            t_c = targets_one_hot[:, c].reshape(targets.shape[0], -1)

            intersection = torch.sum(p_c * t_c, dim=1)
            cardinality = torch.sum(p_c, dim=1) + torch.sum(t_c, dim=1)
            dice_c = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_scores.append(dice_c)

        dice_loss = 1.0 - torch.mean(torch.stack(dice_scores, dim=0))
        total_loss = self.lambda_dice * dice_loss + (1.0 - self.lambda_dice) * ce_loss
        return total_loss
