"""
src/data/label_mapping.py
=========================
Standardized Anatomical Label Harmonization between AMOS 2022 (Source Domain)
and BTCV (Target Domain).

This module guarantees strict anatomical taxonomy alignment across datasets:
- Standardized Unified Index: 13 Overlapping Organ Classes (1 to 13) + Background (0).
- AMOS raw indices (15 classes) -> Unified standard indices.
- BTCV raw indices (13 classes) -> Unified standard indices.
"""

import numpy as np
import torch

# Standardized 13 Common Overlapping Organs + Background
UNIFIED_ORGAN_NAMES = {
    0: "Background",
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava (IVC)",
    10: "Portal & Splenic Vein",
    11: "Pancreas",
    12: "Right Adrenal Gland",
    13: "Left Adrenal Gland"
}

# Raw AMOS 2022 Index -> Unified Standard Index
# Note: AMOS classes 13 (Duodenum), 14 (Bladder), 15 (Prostate/Uterus)
# do not exist in standard BTCV and are mapped to 0 (Background) during cross-domain evaluation.
AMOS_TO_UNIFIED = {
    0: 0,   # Background
    1: 1,   # Spleen -> 1
    2: 2,   # Right Kidney -> 2
    3: 3,   # Left Kidney -> 3
    4: 4,   # Gallbladder -> 4
    5: 5,   # Esophagus -> 5
    6: 6,   # Liver -> 6
    7: 7,   # Stomach -> 7
    8: 8,   # Aorta -> 8
    9: 9,   # IVC -> 9
    10: 11, # Pancreas (AMOS index 10) -> Unified index 11
    11: 12, # Right Adrenal (AMOS index 11) -> Unified index 12
    12: 13, # Left Adrenal (AMOS index 12) -> Unified index 13
    13: 0,  # Duodenum -> Background (Not in BTCV)
    14: 0,  # Bladder -> Background (Not in BTCV)
    15: 0   # Prostate/Uterus -> Background (Not in BTCV)
}

# Raw BTCV Index -> Unified Standard Index
BTCV_TO_UNIFIED = {
    0: 0,   # Background
    1: 1,   # Spleen
    2: 2,   # Right Kidney
    3: 3,   # Left Kidney
    4: 4,   # Gallbladder
    5: 5,   # Esophagus
    6: 6,   # Liver
    7: 7,   # Stomach
    8: 8,   # Aorta
    9: 9,   # Inferior Vena Cava (IVC)
    10: 10, # Portal & Splenic Vein
    11: 11, # Pancreas
    12: 12, # Right Adrenal Gland
    13: 13  # Left Adrenal Gland
}


def remap_labels(label_array: np.ndarray, dataset_type: str = "amos") -> np.ndarray:
    """
    Harmonizes raw label mask arrays into the unified 14-class index space (0-13).
    """
    dataset_type = dataset_type.lower()
    if dataset_type == "amos":
        mapping_dict = AMOS_TO_UNIFIED
    elif dataset_type == "btcv":
        mapping_dict = BTCV_TO_UNIFIED
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Must be 'amos' or 'btcv'.")

    remapped = np.zeros_like(label_array, dtype=np.int64)
    for raw_idx, unified_idx in mapping_dict.items():
        if unified_idx != 0:
            remapped[label_array == raw_idx] = unified_idx

    return remapped


def remap_tensor(label_tensor: torch.Tensor, dataset_type: str = "amos") -> torch.Tensor:
    """
    Tensor version of label remapping for PyTorch tensors.
    """
    dataset_type = dataset_type.lower()
    mapping_dict = AMOS_TO_UNIFIED if dataset_type == "amos" else BTCV_TO_UNIFIED
    remapped = torch.zeros_like(label_tensor, dtype=torch.long)
    for raw_idx, unified_idx in mapping_dict.items():
        if unified_idx != 0:
            remapped[label_tensor == raw_idx] = unified_idx
    return remapped
