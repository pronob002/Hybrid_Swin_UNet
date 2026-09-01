"""
scripts/train_fewshot.py
========================
Few-Shot Cross-Domain Adaptation and Benchmark Evaluation Pipeline.

Adapts models (Hybrid Swin-UNet or 3D U-Net) on k-shot support volumes from AMOS CT (Source)
and evaluates generalization on unseen BTCV CT volumes (Target).
"""

import os
import sys
import time
import argparse
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import CT3DDataset, get_amos_fewshot_cases, get_btcv_case_ids
from src.data.label_mapping import UNIFIED_ORGAN_NAMES
from src.models.hybrid_swin_unet import HybridSwinUNet
from src.models.unet3d import UNet3D
from src.utils.losses import DiceCELoss3D
from src.utils.metrics import evaluate_case


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fewshot_adaptation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cpu"
) -> List[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DiceCELoss3D(num_classes=14)
    loss_history = []

    print(f"\n--- Starting {num_epochs}-Epoch Few-Shot Adaptation on Device: {device} ---")
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        t_epoch = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} (Time: {time.time()-t_epoch:.1f}s)")

    return loss_history


def evaluate_target_domain(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_cases: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    model.eval()
    all_organ_metrics = {name: {"DSC": [], "HD95": [], "ASD": []} for name in list(UNIFIED_ORGAN_NAMES.values())[1:]}
    
    print("\n--- Evaluating on BTCV Target Domain ---")
    case_count = 0
    with torch.no_grad():
        for batch in val_loader:
            case_id = batch["case_id"][0]
            image = batch["image"].to(device)
            label = batch["label"][0].numpy()

            logits = model(image)
            preds = torch.argmax(logits, dim=1)[0].cpu().numpy()

            case_res = evaluate_case(preds, label, num_classes=14)
            for organ, vals in case_res.items():
                all_organ_metrics[organ]["DSC"].append(vals["DSC (%)"])
                all_organ_metrics[organ]["HD95"].append(vals["HD95 (mm)"])
                all_organ_metrics[organ]["ASD"].append(vals["ASD (mm)"])

            case_count += 1
            if max_cases and case_count >= max_cases:
                break

    summary = {}
    for organ, vals in all_organ_metrics.items():
        summary[organ] = {
            "Mean DSC (%)": float(np.mean(vals["DSC"])),
            "Std DSC (%)": float(np.std(vals["DSC"])),
            "Mean HD95 (mm)": float(np.mean(vals["HD95"])),
            "Mean ASD (mm)": float(np.mean(vals["ASD"]))
        }
    return summary


def run_experiment(
    model_type: str = "hybrid_swin",
    k_shots: int = 2,
    num_epochs: int = 5,
    seed: int = 42,
    eval_cases: int = 2
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=================================================================")
    print(f" EXPERIMENT: {model_type.upper()} | {k_shots}-SHOT ADAPTATION | SEED: {seed}")
    print("=================================================================")

    support_cases = get_amos_fewshot_cases(k=k_shots, seed=seed)
    print(f"Sampled {k_shots} AMOS Support Cases: {support_cases}")
    support_ds = CT3DDataset(case_ids=support_cases, dataset_type="amos", is_training=True)
    support_loader = DataLoader(support_ds, batch_size=1, shuffle=True)

    btcv_cases = get_btcv_case_ids()[:eval_cases]
    print(f"Evaluation BTCV Cases ({len(btcv_cases)}): {btcv_cases}")
    target_ds = CT3DDataset(case_ids=btcv_cases, dataset_type="btcv", is_training=False)
    target_loader = DataLoader(target_ds, batch_size=1, shuffle=False)

    if model_type == "hybrid_swin":
        model = HybridSwinUNet(in_channels=1, out_channels=14, embed_dim=24).to(device)
    elif model_type == "unet3d":
        model = UNet3D(in_channels=1, out_channels=14, base_channels=24).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_fewshot_adaptation(model, support_loader, num_epochs=num_epochs, device=device)
    results = evaluate_target_domain(model, target_loader, device=device, max_cases=eval_cases)

    print("\n" + "=" * 80)
    print(f"{'Organ Name':<28} | {'Mean DSC (%)':<15} | {'Mean HD95 (mm)':<15} | {'Mean ASD (mm)':<12}")
    print("-" * 80)
    mean_all_dsc = []
    mean_all_hd = []
    for organ, metrics in results.items():
        dsc_str = f"{metrics['Mean DSC (%)']:.1f} +/- {metrics['Std DSC (%)']:.1f}"
        hd_str = f"{metrics['Mean HD95 (mm)']:.1f}"
        asd_str = f"{metrics['Mean ASD (mm)']:.1f}"
        print(f"{organ:<28} | {dsc_str:<15} | {hd_str:<15} | {asd_str:<12}")
        mean_all_dsc.append(metrics["Mean DSC (%)"])
        mean_all_hd.append(metrics["Mean HD95 (mm)"])

    print("=" * 80)
    print(f"{'OVERALL MEAN (13 ORGANS)':<28} | {np.mean(mean_all_dsc):.2f}%         | {np.mean(mean_all_hd):.2f} mm")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hybrid_swin", choices=["hybrid_swin", "unet3d"])
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_cases", type=int, default=2)
    args = parser.parse_args()

    run_experiment(
        model_type=args.model,
        k_shots=args.shots,
        num_epochs=args.epochs,
        seed=args.seed,
        eval_cases=args.eval_cases
    )
