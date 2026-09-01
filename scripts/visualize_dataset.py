"""
scripts/visualize_dataset.py
============================
Interactive & Visual 3D Dataset Inspector for AMOS 2022 and BTCV.

Features:
1. Downloads and inspects a real 3D CT scan from AMOS / BTCV.
2. Computes and prints exact physical metadata:
   - True matrix dimension (e.g. 512 x 512 x 120)
   - Real voxel spacing (e.g. 0.76 x 0.76 x 5.0 mm³)
   - True HU min/max/mean intensity distribution
   - Exact per-organ voxel counts and physical volume (cm³)
3. Generates high-resolution multi-planar reconstruction (MPR):
   - Axial (Transverse), Coronal (Frontal), and Sagittal (Side) views
   - Multi-organ colored label overlay (13 anatomical structures)
   - Raw HU intensity distribution vs. Clipped/Normalized window
4. Saves publication-grade visual figures to figures/dataset_inspection.png.
"""

import os
import sys
import argparse
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
from huggingface_hub import hf_hub_download

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.label_mapping import UNIFIED_ORGAN_NAMES, remap_labels
from src.data.data_loader import TARGET_SPACING, HU_MIN, HU_MAX, resample_3d, preprocess_ct_volume

# Distinct anatomical color palette for 13 organ classes
ORGAN_COLORS = [
    "#000000",  # 0: Background (Transparent)
    "#e41a1c",  # 1: Spleen (Red)
    "#377eb8",  # 2: Right Kidney (Blue)
    "#4daf4a",  # 3: Left Kidney (Green)
    "#984ea3",  # 4: Gallbladder (Purple)
    "#ff7f00",  # 5: Esophagus (Orange)
    "#ffff33",  # 6: Liver (Yellow)
    "#a65628",  # 7: Stomach (Brown)
    "#f781bf",  # 8: Aorta (Pink)
    "#999999",  # 9: IVC (Grey)
    "#66c2a5",  # 10: Portal & Splenic Vein (Teal)
    "#fc8d62",  # 11: Pancreas (Coral)
    "#8da0cb",  # 12: Right Adrenal (Light Blue)
    "#e78ac3",  # 13: Left Adrenal (Light Pink)
]
CMAP_ORGAN = ListedColormap(ORGAN_COLORS)


def download_case(case_id: str = "amos_0048", dataset_type: str = "amos") -> Tuple[str, str]:
    """Downloads a raw CT scan and label from Hugging Face."""
    if dataset_type == "amos":
        repo_id = "MedOtter/amos22-ct-dataset"
        img_subpath = f"train/imagesTr/{case_id}.nii.gz"
        lbl_subpath = f"train/labelsTr/{case_id}.nii.gz"
    else:
        repo_id = "Live12/btcv"
        img_subpath = f"RawData/Training/img/{case_id}.nii.gz"
        lbl_subpath = f"RawData/Training/label/label{case_id[3:]}.nii.gz"

    print(f"\n>>> [HF-HUB] Downloading {dataset_type.upper()} case: '{case_id}' from {repo_id}...")
    img_path = hf_hub_download(repo_id=repo_id, filename=img_subpath, repo_type="dataset")
    lbl_path = hf_hub_download(repo_id=repo_id, filename=lbl_subpath, repo_type="dataset")
    return img_path, lbl_path


def inspect_and_visualize(
    img_path: str,
    lbl_path: str,
    dataset_type: str = "amos",
    case_name: str = "amos_0048",
    output_png: str = "figures/dataset_inspection.png"
):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # 1. Load NIfTI Headers and Raw Arrays
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    raw_img = img_nii.get_fdata().astype(np.float32)
    raw_lbl = lbl_nii.get_fdata().astype(np.int32)
    spacing = tuple(float(v) for v in img_nii.header.get_zooms()[:3])
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

    print("=" * 75)
    print(f"ANATOMICAL & PHYSICAL METADATA: {case_name.upper()} ({dataset_type.upper()})")
    print("=" * 75)
    print(f"  • True Matrix Dimensions (X x Y x Z) : {raw_img.shape[0]} x {raw_img.shape[1]} x {raw_img.shape[2]} voxels")
    print(f"  • Physical Voxel Spacings (dx, dy, dz): {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm³")
    print(f"  • Total Physical Volume              : {(raw_img.shape[0]*spacing[0]*raw_img.shape[1]*spacing[1]*raw_img.shape[2]*spacing[2])/1000.0:.1f} cm³")
    print(f"  • Raw Hounsfield Unit (HU) Range     : [{np.min(raw_img):.1f}, {np.max(raw_img):.1f}] HU (Mean: {np.mean(raw_img):.1f} HU)")

    # 2. Harmonize Labels to Unified 13 Classes
    remapped_lbl = remap_labels(raw_lbl, dataset_type=dataset_type)

    print("\n  • Per-Organ Physical Breakdown:")
    print(f"    {'ID':<4} | {'Organ Name':<26} | {'Voxel Count':<12} | {'Volume (cm³)':<12}")
    print("    " + "-" * 62)
    for c in range(1, 14):
        organ_name = UNIFIED_ORGAN_NAMES[c]
        v_count = int(np.sum(remapped_lbl == c))
        v_vol = v_count * voxel_volume_cm3
        if v_count > 0:
            print(f"    {c:<4} | {organ_name:<26} | {v_count:<12,d} | {v_vol:<12.2f}")
        else:
            print(f"    {c:<4} | {organ_name:<26} | {'(Absent)':<12} | {'0.00':<12}")

    # 3. Apply Standard Abdominal Preprocessing
    proc_img, proc_lbl = preprocess_ct_volume(raw_img, remapped_lbl, spacing=spacing, dataset_type=dataset_type)

    # 4. Generate Multi-Planar Reconstruction (Axial, Coronal, Sagittal)
    # Find slice with maximum organ diversity
    slice_organ_counts = [len(np.unique(proc_lbl[:, :, z][proc_lbl[:, :, z] > 0])) for z in range(proc_lbl.shape[2])]
    best_z = int(np.argmax(slice_organ_counts))
    best_y = proc_lbl.shape[1] // 2
    best_x = proc_lbl.shape[0] // 2

    fig = plt.figure(figsize=(18, 11), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 0.8], hspace=0.25, wspace=0.2)

    # Axial View (Transverse: Top-Down)
    ax1 = fig.add_subplot(gs[0, 0])
    img_ax = np.rot90(proc_img[:, :, best_z])
    lbl_ax = np.rot90(proc_lbl[:, :, best_z])
    ax1.imshow(img_ax, cmap="gray", origin="upper")
    masked_lbl_ax = np.ma.masked_where(lbl_ax == 0, lbl_ax)
    ax1.imshow(masked_lbl_ax, cmap=CMAP_ORGAN, vmin=0, vmax=13, alpha=0.55, interpolation="nearest")
    ax1.set_title(f"Axial View (Transverse)\nSlice Z={best_z}/{proc_lbl.shape[2]}", fontsize=11, fontweight="bold")
    ax1.axis("off")

    # Coronal View (Frontal: Anterior-Posterior)
    ax2 = fig.add_subplot(gs[0, 1])
    img_cor = np.rot90(proc_img[:, best_y, :])
    lbl_cor = np.rot90(proc_lbl[:, best_y, :])
    ax2.imshow(img_cor, cmap="gray", origin="upper")
    masked_lbl_cor = np.ma.masked_where(lbl_cor == 0, lbl_cor)
    ax2.imshow(masked_lbl_cor, cmap=CMAP_ORGAN, vmin=0, vmax=13, alpha=0.55, interpolation="nearest")
    ax2.set_title(f"Coronal View (Frontal)\nSlice Y={best_y}/{proc_lbl.shape[1]}", fontsize=11, fontweight="bold")
    ax2.axis("off")

    # Sagittal View (Side: Left-Right)
    ax3 = fig.add_subplot(gs[0, 2])
    img_sag = np.rot90(proc_img[best_x, :, :])
    lbl_sag = np.rot90(proc_lbl[best_x, :, :])
    ax3.imshow(img_sag, cmap="gray", origin="upper")
    masked_lbl_sag = np.ma.masked_where(lbl_sag == 0, lbl_sag)
    ax3.imshow(masked_lbl_sag, cmap=CMAP_ORGAN, vmin=0, vmax=13, alpha=0.55, interpolation="nearest")
    ax3.set_title(f"Sagittal View (Side)\nSlice X={best_x}/{proc_lbl.shape[0]}", fontsize=11, fontweight="bold")
    ax3.axis("off")

    # HU Intensity Distribution Plot (Raw vs. Windowed)
    ax_hist = fig.add_subplot(gs[1, 0:2])
    raw_sample = raw_img.flatten()[::20]  # Subsample for speed
    ax_hist.hist(raw_sample, bins=100, range=(-1000, 1000), color="#4a7bb0", alpha=0.7, label="Raw CT HU Distribution")
    ax_hist.axvspan(-125, 275, color="#e74c3c", alpha=0.25, label=f"Abdominal Window [{HU_MIN:.0f}, {HU_MAX:.0f}] HU")
    ax_hist.set_title("CT Hounsfield Unit (HU) Scale & Abdominal Windowing", fontsize=11, fontweight="bold")
    ax_hist.set_xlabel("Hounsfield Units (HU)", fontsize=10)
    ax_hist.set_ylabel("Voxel Count Frequency", fontsize=10)
    ax_hist.grid(True, linestyle=":", alpha=0.5)
    ax_hist.legend(frameon=True, loc="upper right")

    # Legend Table for 13 Organs
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.axis("off")
    for i in range(1, 14):
        color = ORGAN_COLORS[i]
        name = UNIFIED_ORGAN_NAMES[i]
        v_count = int(np.sum(remapped_lbl == i))
        ax_leg.scatter([0.05], [1.0 - (i * 0.07)], color=color, s=120, edgecolors="black", linewidths=0.8)
        ax_leg.text(0.15, 1.0 - (i * 0.07), f"{i}. {name} ({v_count:,} voxels)", fontsize=8.5, verticalalignment="center")
    ax_leg.set_title("13 Unified Organ Taxonomy", fontsize=11, fontweight="bold", pad=10)
    ax_leg.set_xlim(0, 1.0)
    ax_leg.set_ylim(0, 1.05)

    plt.suptitle(
        f"3D CT Dataset Inspection & Physical Reconstruction: {case_name.upper()} ({dataset_type.upper()})\n"
        f"Original Spacing: {spacing[0]:.2f}x{spacing[1]:.2f}x{spacing[2]:.2f} mm³ | Resampled Isotropic: {TARGET_SPACING[0]}x{TARGET_SPACING[1]}x{TARGET_SPACING[2]} mm³",
        fontsize=13,
        fontweight="bold",
        y=0.98
    )

    plt.savefig(output_png, bbox_inches="tight")
    plt.close()
    print(f"\n>>> [FIGURE] Successfully rendered and saved 3D dataset inspection figure to '{output_png}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Multi-Organ Dataset Visualizer & Inspector")
    parser.add_argument("--case_id", type=str, default="amos_0048", help="Case ID to inspect (e.g. amos_0048, img0001)")
    parser.add_argument("--dataset", type=str, default="amos", choices=["amos", "btcv"], help="Dataset domain ('amos' or 'btcv')")
    parser.add_argument("--output", type=str, default="figures/dataset_inspection.png", help="Output PNG path")
    args = parser.parse_args()

    img_p, lbl_p = download_case(case_id=args.case_id, dataset_type=args.dataset)
    inspect_and_visualize(img_p, lbl_p, dataset_type=args.dataset, case_name=args.case_id, output_png=args.output)
