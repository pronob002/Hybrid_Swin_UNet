"""
src/data/data_loader.py
=======================
Memory-Efficient Data Pipeline for 3D Abdominal Multi-Organ Segmentation.
Supports On-Demand Hugging Face Streaming, Isotropic Resampling, HU Windowing,
Taxonomy Harmonization, and 3D Patch Extraction.
"""

import os
import random
from typing import List, Tuple, Optional, Dict
import numpy as np
import scipy.ndimage as ndimage
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

try:
    from .label_mapping import remap_labels, UNIFIED_ORGAN_NAMES
except ImportError:
    from label_mapping import remap_labels, UNIFIED_ORGAN_NAMES

# Target Spacing & Intensity Hyperparameters
TARGET_SPACING = (1.5, 1.5, 1.5)  # mm isotropic
HU_MIN = -125.0
HU_MAX = 275.0
PATCH_SIZE = (96, 96, 96)  # Depth, Height, Width (D, H, W)


def resample_3d(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = TARGET_SPACING,
    is_label: bool = False
) -> np.ndarray:
    """
    Resamples a 3D volume to target isotropic voxel spacing.
    """
    zoom_factors = [
        float(cur) / float(tgt)
        for cur, tgt in zip(current_spacing, target_spacing)
    ]
    order = 0 if is_label else 1  # 0: Nearest Neighbor (discrete labels), 1: Linear interpolation
    resampled = ndimage.zoom(volume, zoom=zoom_factors, order=order, mode="nearest")
    return resampled


def preprocess_ct_volume(
    image_data: np.ndarray,
    label_data: Optional[np.ndarray],
    spacing: Tuple[float, float, float],
    dataset_type: str = "amos"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Standardizes a raw CT scan and label mask:
    1. Harmonizes labels using label_mapping.
    2. Resamples image and label to isotropic 1.5 mm spacing.
    3. Clips HU intensity to [-125, +275].
    4. Z-score normalizes intensity (mean=0, std=1).
    """
    if label_data is not None:
        label_data = remap_labels(label_data.astype(np.int32), dataset_type=dataset_type)

    resampled_img = resample_3d(image_data.astype(np.float32), current_spacing=spacing, is_label=False)
    resampled_lbl = None
    if label_data is not None:
        resampled_lbl = resample_3d(label_data, current_spacing=spacing, is_label=True)

    clipped_img = np.clip(resampled_img, HU_MIN, HU_MAX)

    mean_val = np.mean(clipped_img)
    std_val = np.std(clipped_img) + 1e-8
    norm_img = (clipped_img - mean_val) / std_val

    return norm_img, resampled_lbl


def extract_3d_patch(
    image: np.ndarray,
    label: Optional[np.ndarray],
    patch_size: Tuple[int, int, int] = PATCH_SIZE,
    foreground_bias: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extracts a cubic 3D patch of shape (96, 96, 96).
    """
    shape = image.shape
    pad_needed = [max(0, patch_size[i] - shape[i]) for i in range(3)]
    
    if any(p > 0 for p in pad_needed):
        pad_width = [(p // 2, p - p // 2) for p in pad_needed]
        image = np.pad(image, pad_width, mode="constant", constant_values=0)
        if label is not None:
            label = np.pad(label, pad_width, mode="constant", constant_values=0)
        shape = image.shape

    max_x = shape[0] - patch_size[0]
    max_y = shape[1] - patch_size[1]
    max_z = shape[2] - patch_size[2]

    if foreground_bias and label is not None and np.any(label > 0):
        if random.random() < 0.7:
            fg_indices = np.argwhere(label > 0)
            center = fg_indices[random.randint(0, len(fg_indices) - 1)]
            start_x = np.clip(center[0] - patch_size[0] // 2, 0, max_x)
            start_y = np.clip(center[1] - patch_size[1] // 2, 0, max_y)
            start_z = np.clip(center[2] - patch_size[2] // 2, 0, max_z)
        else:
            start_x = random.randint(0, max_x)
            start_y = random.randint(0, max_y)
            start_z = random.randint(0, max_z)
    else:
        start_x = random.randint(0, max_x) if max_x > 0 else 0
        start_y = random.randint(0, max_y) if max_y > 0 else 0
        start_z = random.randint(0, max_z) if max_z > 0 else 0

    img_patch = image[
        start_x : start_x + patch_size[0],
        start_y : start_y + patch_size[1],
        start_z : start_z + patch_size[2]
    ]

    lbl_patch = None
    if label is not None:
        lbl_patch = label[
            start_x : start_x + patch_size[0],
            start_y : start_y + patch_size[1],
            start_z : start_z + patch_size[2]
        ]

    return img_patch, lbl_patch


class CT3DDataset(Dataset):
    """
    PyTorch Dataset for on-demand 3D volumetric CT patches from Hugging Face.
    """
    def __init__(
        self,
        case_ids: List[str],
        dataset_type: str = "btcv",
        is_training: bool = True,
        patch_size: Tuple[int, int, int] = PATCH_SIZE
    ):
        self.case_ids = case_ids
        self.dataset_type = dataset_type.lower()
        self.is_training = is_training
        self.patch_size = patch_size

        if self.dataset_type == "amos":
            self.repo_id = "MedOtter/amos22-ct-dataset"
        elif self.dataset_type == "btcv":
            self.repo_id = "Live12/btcv"
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    def __len__(self) -> int:
        return len(self.case_ids)

    def _get_file_paths(self, case_id: str) -> Tuple[str, str]:
        if self.dataset_type == "amos":
            img_rel = f"train/imagesTr/{case_id}.nii.gz"
            lbl_rel = f"train/labelsTr/{case_id}.nii.gz"
        else:  # btcv
            img_rel = f"RawData/Training/img/{case_id}.nii.gz"
            lbl_rel = f"RawData/Training/label/label{case_id[3:]}.nii.gz"

        img_path = hf_hub_download(repo_id=self.repo_id, filename=img_rel, repo_type="dataset")
        lbl_path = hf_hub_download(repo_id=self.repo_id, filename=lbl_rel, repo_type="dataset")
        return img_path, lbl_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[idx]
        img_path, lbl_path = self._get_file_paths(case_id)

        img_nii = nib.load(img_path)
        lbl_nii = nib.load(lbl_path)
        
        spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])
        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata().astype(np.int32)

        norm_img, harm_lbl = preprocess_ct_volume(
            img_data, lbl_data, spacing=spacing, dataset_type=self.dataset_type
        )

        img_patch, lbl_patch = extract_3d_patch(
            norm_img, harm_lbl, patch_size=self.patch_size, foreground_bias=self.is_training
        )

        if self.is_training:
            if random.random() < 0.5:
                img_patch = np.flip(img_patch, axis=0).copy()
                lbl_patch = np.flip(lbl_patch, axis=0).copy()
            if random.random() < 0.5:
                img_patch = np.flip(img_patch, axis=1).copy()
                lbl_patch = np.flip(lbl_patch, axis=1).copy()
            if random.random() < 0.3:
                scale_factor = random.uniform(0.9, 1.1)
                img_patch = img_patch * scale_factor

        tensor_img = torch.from_numpy(img_patch).float().unsqueeze(0)  # (1, D, H, W)
        tensor_lbl = torch.from_numpy(lbl_patch).long()                # (D, H, W)

        return {
            "image": tensor_img,
            "label": tensor_lbl,
            "case_id": case_id
        }


def get_btcv_case_ids() -> List[str]:
    """Returns 30 BTCV target evaluation case IDs."""
    ids = [f"img{i:04d}" for i in range(1, 11)] + [f"img{i:04d}" for i in range(21, 41)]
    return ids


def get_amos_fewshot_cases(k: int = 5, seed: int = 42) -> List[str]:
    """Selects k random support scan IDs from AMOS CT training set."""
    random.seed(seed)
    amos_pool = [
        "amos_0001", "amos_0004", "amos_0005", "amos_0006", "amos_0007",
        "amos_0009", "amos_0010", "amos_0011", "amos_0014", "amos_0015",
        "amos_0016", "amos_0017", "amos_0018", "amos_0019", "amos_0020"
    ]
    sampled = random.sample(amos_pool, min(k, len(amos_pool)))
    return sampled
