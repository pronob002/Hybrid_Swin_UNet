"""
src/utils/metrics.py
====================
Evaluation Metrics for 3D Multi-Organ Segmentation.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.ndimage as ndimage
import torch

try:
    from src.data.label_mapping import UNIFIED_ORGAN_NAMES
except ImportError:
    from label_mapping import UNIFIED_ORGAN_NAMES


def compute_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int = 14
) -> Dict[int, float]:
    """Computes per-class Dice Similarity Coefficient (DSC in %)."""
    dice_results = {}
    for c in range(1, num_classes):
        p_c = (pred_mask == c)
        g_c = (gt_mask == c)

        intersection = np.logical_and(p_c, g_c).sum()
        total = p_c.sum() + g_c.sum()

        if total == 0:
            dice = 100.0 if np.array_equal(p_c, g_c) else 0.0
        else:
            dice = (2.0 * intersection / total) * 100.0

        dice_results[c] = float(dice)

    return dice_results


def compute_surface_distances(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> Tuple[float, float]:
    """Computes HD95 and ASD between two binary 3D masks in millimeters."""
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 50.0, 20.0

    pred_border = pred_bin ^ ndimage.binary_erosion(pred_bin)
    gt_border = gt_bin ^ ndimage.binary_erosion(gt_bin)

    dt_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = ndimage.distance_transform_edt(~gt_border, sampling=spacing)

    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    if len(all_distances) == 0:
        return 0.0, 0.0

    hd95 = float(np.percentile(all_distances, 95))
    asd = float(np.mean(all_distances))

    return hd95, asd


def evaluate_case(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = 14,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> Dict[str, Dict[str, float]]:
    """Evaluates a full 3D case, returning per-organ DSC, HD95, and ASD."""
    results = {}
    dice_dict = compute_dice_score(pred, gt, num_classes)

    for c in range(1, num_classes):
        organ_name = UNIFIED_ORGAN_NAMES.get(c, f"Class_{c}")
        p_c = (pred == c)
        g_c = (gt == c)

        if g_c.sum() > 0:
            hd95, asd = compute_surface_distances(p_c, g_c, spacing)
        else:
            hd95, asd = 0.0, 0.0

        results[organ_name] = {
            "DSC (%)": dice_dict[c],
            "HD95 (mm)": hd95,
            "ASD (mm)": asd
        }

    return results
