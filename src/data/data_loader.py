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


# 200 Verified AMOS 2022 CT Training Cases on Hugging Face (MedOtter/amos22-ct-dataset)
VERIFIED_AMOS_CASES = [
    "amos_0001", "amos_0004", "amos_0005", "amos_0006", "amos_0007",
    "amos_0009", "amos_0010", "amos_0011", "amos_0014", "amos_0015",
    "amos_0016", "amos_0017", "amos_0019", "amos_0021", "amos_0023",
    "amos_0024", "amos_0025", "amos_0027", "amos_0030", "amos_0033",
    "amos_0035", "amos_0036", "amos_0038", "amos_0042", "amos_0043",
    "amos_0044", "amos_0045", "amos_0047", "amos_0048", "amos_0049",
    "amos_0050", "amos_0052", "amos_0054", "amos_0057", "amos_0058",
    "amos_0059", "amos_0060", "amos_0064", "amos_0066", "amos_0067",
    "amos_0069", "amos_0071", "amos_0072", "amos_0075", "amos_0076",
    "amos_0077", "amos_0078", "amos_0079", "amos_0081", "amos_0083",
    "amos_0084", "amos_0086", "amos_0088", "amos_0089", "amos_0092",
    "amos_0094", "amos_0097", "amos_0098", "amos_0099", "amos_0102",
    "amos_0103", "amos_0104", "amos_0105", "amos_0109", "amos_0110",
    "amos_0111", "amos_0113", "amos_0115", "amos_0116", "amos_0118",
    "amos_0119", "amos_0121", "amos_0124", "amos_0125", "amos_0126",
    "amos_0127", "amos_0129", "amos_0131", "amos_0133", "amos_0134",
    "amos_0135", "amos_0137", "amos_0138", "amos_0141", "amos_0142",
    "amos_0143", "amos_0147", "amos_0149", "amos_0152", "amos_0153",
    "amos_0154", "amos_0156", "amos_0158", "amos_0159", "amos_0160",
    "amos_0161", "amos_0162", "amos_0166", "amos_0170", "amos_0171",
    "amos_0172", "amos_0173", "amos_0175", "amos_0177", "amos_0179",
    "amos_0180", "amos_0181", "amos_0184", "amos_0185", "amos_0186",
    "amos_0188", "amos_0190", "amos_0192", "amos_0193", "amos_0195",
    "amos_0196", "amos_0197", "amos_0198", "amos_0199", "amos_0212",
    "amos_0214", "amos_0215", "amos_0217", "amos_0224", "amos_0225",
    "amos_0226", "amos_0230", "amos_0231", "amos_0235", "amos_0237",
    "amos_0239", "amos_0242", "amos_0245", "amos_0248", "amos_0249",
    "amos_0254", "amos_0259", "amos_0263", "amos_0264", "amos_0268",
    "amos_0272", "amos_0273", "amos_0274", "amos_0276", "amos_0279",
    "amos_0281", "amos_0282", "amos_0288", "amos_0294", "amos_0296",
    "amos_0297", "amos_0299", "amos_0301", "amos_0302", "amos_0307",
    "amos_0317", "amos_0320", "amos_0321", "amos_0330", "amos_0332",
    "amos_0336", "amos_0337", "amos_0341", "amos_0348", "amos_0349",
    "amos_0350", "amos_0351", "amos_0353", "amos_0358", "amos_0361",
    "amos_0362", "amos_0366", "amos_0367", "amos_0370", "amos_0371",
    "amos_0374", "amos_0376", "amos_0378", "amos_0379", "amos_0380",
    "amos_0381", "amos_0383", "amos_0384", "amos_0387", "amos_0388",
    "amos_0390", "amos_0391", "amos_0392", "amos_0395", "amos_0396",
    "amos_0398", "amos_0400", "amos_0401", "amos_0402", "amos_0403",
    "amos_0404", "amos_0405", "amos_0406", "amos_0408", "amos_0410"
]


def get_amos_fewshot_cases(k: int = 5, seed: int = 42) -> List[str]:
    """Selects k random support scan IDs from the verified AMOS CT training set."""
    random.seed(seed)
    sampled = random.sample(VERIFIED_AMOS_CASES, min(k, len(VERIFIED_AMOS_CASES)))
    return sampled
