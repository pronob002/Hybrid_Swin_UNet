"""
scripts/train_fewshot.py
========================
Fault-Tolerant Few-Shot Cross-Domain Adaptation & Benchmark Evaluation Pipeline.

Key Features:
1. Automated Per-Epoch Checkpointing (checkpoint_latest.pth & checkpoint_best.pth).
2. Comprehensive Downstream Artifacts: Saves benchmark_summary.csv, benchmark_summary.json,
   per_case_details.json (for Wilcoxon p-value statistical tests), and training_curves.json.
3. Hugging Face Hub Auto-Sync: Automatically pushes all checkpoints & result artifacts to
   your private Hugging Face repository for seamless multi-session resume and analysis.
"""

import os
import sys
import time
import json
import csv
import argparse
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, hf_hub_download, create_repo

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


# ==============================================================================
# Hugging Face Hub Cloud Storage & Sync Utilities
# ==============================================================================

def init_hf_sync(repo_id: Optional[str], token: Optional[str] = None) -> Optional[HfApi]:
    """Initializes and creates the remote Hugging Face repository if not existing."""
    if not repo_id:
        return None
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️ [HF-SYNC] Warning: Hugging Face repo specified but no HF_TOKEN found. Cloud upload disabled.")
        return None

    try:
        api = HfApi(token=token)
        create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True, private=True)
        print(f">>> [HF-SYNC] Cloud repository ready: https://huggingface.co/{repo_id} (Private)")
        return api
    except Exception as e:
        print(f"⚠️ [HF-SYNC] Failed to initialize HF repo '{repo_id}': {e}")
        return None


def pull_file_from_hf(
    repo_id: Optional[str],
    filename_in_repo: str,
    local_target_path: str,
    token: Optional[str] = None
) -> bool:
    """Pulls a single checkpoint or result file from Hugging Face Hub if available."""
    if not repo_id:
        return False
    token = token or os.environ.get("HF_TOKEN")
    try:
        os.makedirs(os.path.dirname(local_target_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename_in_repo,
            repo_type="model",
            token=token,
            local_dir=os.path.dirname(local_target_path)
        )
        print(f">>> [HF-SYNC] Successfully pulled '{filename_in_repo}' from Hugging Face Hub.")
        return True
    except Exception:
        # File doesn't exist yet on remote repo (e.g. fresh run)
        return False


def push_file_to_hf(
    api: Optional[HfApi],
    repo_id: Optional[str],
    local_path: str,
    path_in_repo: str,
    token: Optional[str] = None
):
    """Pushes a local file to Hugging Face Hub."""
    if not api or not repo_id or not os.path.isfile(local_path):
        return
    token = token or os.environ.get("HF_TOKEN")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f">>> [HF-SYNC] Uploaded '{path_in_repo}' to Hugging Face Hub.")
    except Exception as e:
        print(f"⚠️ [HF-SYNC] Upload failed for '{path_in_repo}': {e}")


# ==============================================================================
# Checkpointing & Auto-Resume Logic
# ==============================================================================

def get_checkpoint_dir(base_dir: str, model_type: str, k_shots: int, seed: int) -> str:
    path = os.path.join(base_dir, f"{model_type}_k{k_shots}_seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    hf_api: Optional[HfApi] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    exp_subpath: Optional[str] = None
):
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    torch.save(state, latest_path)

    best_path = None
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        torch.save(state, best_path)

    # Cloud sync to Hugging Face Hub
    if hf_api and hf_repo and exp_subpath:
        push_file_to_hf(
            hf_api, hf_repo,
            local_path=latest_path,
            path_in_repo=f"checkpoints/{exp_subpath}/checkpoint_latest.pth",
            token=hf_token
        )
        if is_best and best_path:
            push_file_to_hf(
                hf_api, hf_repo,
                local_path=best_path,
                path_in_repo=f"checkpoints/{exp_subpath}/checkpoint_best.pth",
                token=hf_token
            )


def load_checkpoint_if_exists(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    exp_subpath: Optional[str] = None
) -> Tuple[int, float, List[float]]:
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

    # If local file missing, attempt to pull from Hugging Face Hub
    if not os.path.isfile(latest_path) and hf_repo and exp_subpath:
        pull_file_from_hf(
            repo_id=hf_repo,
            filename_in_repo=f"checkpoints/{exp_subpath}/checkpoint_latest.pth",
            local_target_path=latest_path,
            token=hf_token
        )
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        pull_file_from_hf(
            repo_id=hf_repo,
            filename_in_repo=f"checkpoints/{exp_subpath}/checkpoint_best.pth",
            local_target_path=best_path,
            token=hf_token
        )

    if not os.path.isfile(latest_path):
        return 1, float("inf"), []

    print(f"\n>>> [AUTO-RESUME] Found existing checkpoint: {latest_path}")
    checkpoint = torch.load(latest_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    loss_history = checkpoint.get("loss_history", [])

    print(f">>> [AUTO-RESUME] Successfully restored state from Epoch {checkpoint['epoch']} (Best Loss: {best_loss:.4f})")
    print(f">>> [AUTO-RESUME] Resuming training from Epoch {start_epoch}...\n")
    return start_epoch, best_loss, loss_history


def train_fewshot_adaptation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    checkpoint_dir: Optional[str] = None,
    hf_api: Optional[HfApi] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    exp_subpath: Optional[str] = None
) -> List[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DiceCELoss3D(num_classes=14)

    start_epoch = 1
    best_loss = float("inf")
    loss_history: List[float] = []

    if checkpoint_dir:
        start_epoch, best_loss, loss_history = load_checkpoint_if_exists(
            checkpoint_dir, model, optimizer, device=device,
            hf_repo=hf_repo, hf_token=hf_token, exp_subpath=exp_subpath
        )

    if start_epoch > num_epochs:
        print(f">>> [COMPLETED] Model has already completed all {num_epochs} epochs. Skipping training.")
        if checkpoint_dir:
            best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
            if os.path.isfile(best_path):
                best_ckpt = torch.load(best_path, map_location=device)
                model.load_state_dict(best_ckpt["model_state_dict"])
                print(f">>> Loaded best checkpoint weights from {best_path}")
        return loss_history

    print(f"\n--- Starting Few-Shot Adaptation (Epochs {start_epoch} -> {num_epochs}) on Device: {device} ---")
    for epoch in range(start_epoch, num_epochs + 1):
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
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] Loss: {avg_loss:.4f} {'[BEST]' if is_best else ''} (Time: {time.time()-t_epoch:.1f}s)")

        # Save checkpoint locally and upload to Hugging Face Hub
        if checkpoint_dir:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "loss_history": loss_history
            }
            save_checkpoint(
                state,
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
                hf_api=hf_api,
                hf_repo=hf_repo,
                hf_token=hf_token,
                exp_subpath=exp_subpath
            )

    # Load best model weights for subsequent evaluation
    if checkpoint_dir:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        if os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            print(f"\n>>> Loaded best model weights from {best_path} for evaluation.")

    return loss_history


def evaluate_target_domain(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str = "cpu",
    max_cases: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Evaluates on unseen BTCV cases. Returns:
    1. Summary dictionary across all cases (Mean/Std DSC, Mean HD95, Mean ASD per organ).
    2. Detailed per-case dictionary for statistical significance testing (Wilcoxon signed-rank).
    """
    model.eval()
    all_organ_metrics = {name: {"DSC": [], "HD95": [], "ASD": []} for name in list(UNIFIED_ORGAN_NAMES.values())[1:]}
    per_case_details: Dict[str, Dict[str, Dict[str, float]]] = {}

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
            per_case_details[case_id] = case_res

            for organ, vals in case_res.items():
                all_organ_metrics[organ]["DSC"].append(vals["DSC (%)"])
                all_organ_metrics[organ]["HD95"].append(vals["HD95 (mm)"])
                all_organ_metrics[organ]["ASD"].append(vals["ASD (mm)"])

            case_count += 1
            print(f"  Processed Case [{case_count}/{max_cases or len(val_loader)}]: {case_id}")
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
    return summary, per_case_details


def update_results_cache(
    results_dir: str,
    exp_key: str,
    exp_data: Dict[str, Any],
    hf_api: Optional[HfApi] = None,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None
):
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "benchmark_summary.json")
    csv_path = os.path.join(results_dir, "benchmark_summary.csv")
    per_case_path = os.path.join(results_dir, "per_case_details.json")
    training_curves_path = os.path.join(results_dir, "training_curves.json")

    # 1. Update benchmark_summary.json
    cached = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                cached = json.load(f)
        except Exception:
            cached = {}
    cached[exp_key] = exp_data
    with open(json_path, "w") as f:
        json.dump(cached, f, indent=2)

    # 2. Update benchmark_summary.csv
    fieldnames = [
        "Experiment_Key", "Model", "Shots", "Seed", "Epochs",
        "Mean_DSC", "Mean_HD95", "Mean_ASD"
    ] + [f"{name}_DSC" for name in list(UNIFIED_ORGAN_NAMES.values())[1:]]

    rows = []
    for k, v in cached.items():
        row = {
            "Experiment_Key": k,
            "Model": v.get("model", ""),
            "Shots": v.get("shots", ""),
            "Seed": v.get("seed", ""),
            "Epochs": v.get("epochs", ""),
            "Mean_DSC": round(v.get("overall_mean_dsc", 0.0), 2),
            "Mean_HD95": round(v.get("overall_mean_hd95", 0.0), 2),
            "Mean_ASD": round(v.get("overall_mean_asd", 0.0), 2),
        }
        per_organ = v.get("per_organ", {})
        for name in list(UNIFIED_ORGAN_NAMES.values())[1:]:
            row[f"{name}_DSC"] = round(per_organ.get(name, {}).get("Mean DSC (%)", 0.0), 2)
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 3. Update per_case_details.json (for Wilcoxon statistical testing)
    cached_cases = {}
    if os.path.isfile(per_case_path):
        try:
            with open(per_case_path, "r") as f:
                cached_cases = json.load(f)
        except Exception:
            cached_cases = {}
    cached_cases[exp_key] = exp_data.get("per_case_details", {})
    with open(per_case_path, "w") as f:
        json.dump(cached_cases, f, indent=2)

    # 4. Update training_curves.json (for convergence figures)
    cached_curves = {}
    if os.path.isfile(training_curves_path):
        try:
            with open(training_curves_path, "r") as f:
                cached_curves = json.load(f)
        except Exception:
            cached_curves = {}
    cached_curves[exp_key] = exp_data.get("loss_history", [])
    with open(training_curves_path, "w") as f:
        json.dump(cached_curves, f, indent=2)

    print(f"\n>>> [SAVED] Benchmark artifacts cached to {results_dir}/ (summary JSON, CSV, per-case details, loss curves)")

    # Push updated results to Hugging Face Hub
    if hf_api and hf_repo:
        push_file_to_hf(hf_api, hf_repo, local_path=json_path, path_in_repo="results/benchmark_summary.json", token=hf_token)
        push_file_to_hf(hf_api, hf_repo, local_path=csv_path, path_in_repo="results/benchmark_summary.csv", token=hf_token)
        push_file_to_hf(hf_api, hf_repo, local_path=per_case_path, path_in_repo="results/per_case_details.json", token=hf_token)
        push_file_to_hf(hf_api, hf_repo, local_path=training_curves_path, path_in_repo="results/training_curves.json", token=hf_token)


def is_experiment_already_done(
    results_dir: str,
    exp_key: str,
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    json_path = os.path.join(results_dir, "benchmark_summary.json")

    # If local cache missing, check Hugging Face Hub
    if not os.path.isfile(json_path) and hf_repo:
        pull_file_from_hf(
            repo_id=hf_repo,
            filename_in_repo="results/benchmark_summary.json",
            local_target_path=json_path,
            token=hf_token
        )

    if os.path.isfile(json_path):
        try:
            with open(json_path, "r") as f:
                cached = json.load(f)
            if exp_key in cached:
                return cached[exp_key]
        except Exception:
            return None
    return None


def run_experiment(
    model_type: str = "hybrid_swin",
    k_shots: int = 5,
    num_epochs: int = 50,
    seed: int = 42,
    eval_cases: int = 10,
    checkpoint_base_dir: str = "checkpoints",
    results_dir: str = "results",
    hf_repo: Optional[str] = None,
    hf_token: Optional[str] = None,
    force_rerun: bool = False
) -> Dict[str, Any]:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_key = f"{model_type}_k{k_shots}_seed{seed}_ep{num_epochs}"
    exp_subpath = f"{model_type}_k{k_shots}_seed{seed}"

    print("=" * 80)
    print(f" EXPERIMENT: {model_type.upper()} | {k_shots}-SHOT ADAPTATION | SEED: {seed} | DEVICE: {device}")
    if hf_repo:
        print(f" HF CLOUD SYNC: https://huggingface.co/{hf_repo}")
    print("=" * 80)

    # Initialize Hugging Face API connection if specified
    hf_api = init_hf_sync(repo_id=hf_repo, token=hf_token)

    # Check cache to avoid duplicate runs
    if not force_rerun:
        cached_result = is_experiment_already_done(results_dir, exp_key, hf_repo=hf_repo, hf_token=hf_token)
        if cached_result is not None:
            print(f">>> [CACHE HIT] Experiment '{exp_key}' already completed with Mean DSC: {cached_result['overall_mean_dsc']:.2f}%.")
            print(">>> Skipping training and evaluation to save compute time! (Pass --force_rerun to override)\n")
            return cached_result

    ckpt_dir = get_checkpoint_dir(checkpoint_base_dir, model_type, k_shots, seed)

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

    # Train with auto-checkpointing, auto-resume, and Hugging Face Hub cloud sync
    loss_history = train_fewshot_adaptation(
        model,
        support_loader,
        num_epochs=num_epochs,
        device=device,
        checkpoint_dir=ckpt_dir,
        hf_api=hf_api,
        hf_repo=hf_repo,
        hf_token=hf_token,
        exp_subpath=exp_subpath
    )

    # Evaluate
    results, per_case_details = evaluate_target_domain(model, target_loader, device=device, max_cases=eval_cases)

    mean_all_dsc = [m["Mean DSC (%)"] for m in results.values()]
    mean_all_hd = [m["Mean HD95 (mm)"] for m in results.values()]
    mean_all_asd = [m["Mean ASD (mm)"] for m in results.values()]

    print("\n" + "=" * 80)
    print(f"{'Organ Name':<28} | {'Mean DSC (%)':<15} | {'Mean HD95 (mm)':<15} | {'Mean ASD (mm)':<12}")
    print("-" * 80)
    for organ, metrics in results.items():
        dsc_str = f"{metrics['Mean DSC (%)']:.1f} +/- {metrics['Std DSC (%)']:.1f}"
        hd_str = f"{metrics['Mean HD95 (mm)']:.1f}"
        asd_str = f"{metrics['Mean ASD (mm)']:.1f}"
        print(f"{organ:<28} | {dsc_str:<15} | {hd_str:<15} | {asd_str:<12}")

    overall_dsc = float(np.mean(mean_all_dsc))
    overall_hd = float(np.mean(mean_all_hd))
    overall_asd = float(np.mean(mean_all_asd))

    print("=" * 80)
    print(f"{'OVERALL MEAN (13 ORGANS)':<28} | {overall_dsc:.2f}%         | {overall_hd:.2f} mm")
    print("=" * 80)

    exp_payload = {
        "model": model_type,
        "shots": k_shots,
        "seed": seed,
        "epochs": num_epochs,
        "overall_mean_dsc": overall_dsc,
        "overall_mean_hd95": overall_hd,
        "overall_mean_asd": overall_asd,
        "per_organ": results,
        "per_case_details": per_case_details,
        "loss_history": loss_history
    }

    update_results_cache(
        results_dir=results_dir,
        exp_key=exp_key,
        exp_data=exp_payload,
        hf_api=hf_api,
        hf_repo=hf_repo,
        hf_token=hf_token
    )
    return exp_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fault-Tolerant Few-Shot Segmentation Runner with Hugging Face Sync")
    parser.add_argument("--model", type=str, default="hybrid_swin", choices=["hybrid_swin", "unet3d"])
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_cases", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Hugging Face Model Repo ID (e.g. 'Pronob002/hybrid-swin-unet-checkpoints')")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face Access Token (or set HF_TOKEN env var)")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    run_experiment(
        model_type=args.model,
        k_shots=args.shots,
        num_epochs=args.epochs,
        seed=args.seed,
        eval_cases=args.eval_cases,
        checkpoint_base_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        force_rerun=args.force_rerun
    )
