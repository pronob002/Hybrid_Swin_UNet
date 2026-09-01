"""
scripts/analyze_results.py
==========================
Post-Benchmark Analysis & Publication Artifact Generator for Computers in Biology and Medicine (CBM).

This script reads:
1. results/benchmark_summary.json & .csv
2. results/per_case_details.json (for non-parametric Wilcoxon signed-rank p-value tests)
3. results/training_curves.json (for training convergence curves)

Outputs:
1. Publication-ready Markdown & LaTeX tables for the manuscript (Tables III, IV).
2. High-resolution figures (Figure 4: Adaptation curves, Figure 5: Convergence curves) saved to figures/.
"""

import os
import sys
import json
import csv
from typing import Dict, List, Any
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.label_mapping import UNIFIED_ORGAN_NAMES


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def compute_wilcoxon_pvalues(per_case_details: Dict[str, Any], model_a: str = "hybrid_swin", model_b: str = "unet3d", k: int = 5, seed: int = 42, ep: int = 50) -> Dict[str, float]:
    """Computes Wilcoxon signed-rank p-value between model_a and model_b across test cases."""
    key_a = f"{model_a}_k{k}_seed{seed}_ep{ep}"
    key_b = f"{model_b}_k{k}_seed{seed}_ep{ep}"

    cases_a = per_case_details.get(key_a, {})
    cases_b = per_case_details.get(key_b, {})

    common_cases = sorted(list(set(cases_a.keys()).intersection(set(cases_b.keys()))))
    if not common_cases:
        return {}

    p_values = {}
    for organ in list(UNIFIED_ORGAN_NAMES.values())[1:]:
        scores_a = [cases_a[c][organ]["DSC (%)"] for c in common_cases if organ in cases_a[c]]
        scores_b = [cases_b[c][organ]["DSC (%)"] for c in common_cases if organ in cases_b[c]]

        if len(scores_a) >= 5 and not np.allclose(scores_a, scores_b):
            try:
                stat, p_val = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")
                p_values[organ] = float(p_val)
            except Exception:
                p_values[organ] = 1.0
        else:
            p_values[organ] = 1.0

    return p_values


def generate_cbm_tables(summary: Dict[str, Any], per_case: Dict[str, Any], output_md_path: str = "results/cbm_tables.md"):
    """Generates complete publication tables for all 13 organs and multi-regime adaptations."""
    os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
    p_vals = compute_wilcoxon_pvalues(per_case, model_a="hybrid_swin", model_b="unet3d", k=5, seed=42)

    with open(output_md_path, "w") as f:
        f.write("# Empirical Benchmark Results for Computers in Biology and Medicine (CBM)\n\n")
        
        # Table III: 13-Organ Comparison
        f.write("### TABLE III: Per-Organ Cross-Domain Evaluation (13 Anatomical Organs, 5-Shot Adaptation)\n\n")
        f.write("| Unified ID | Anatomical Organ | 3D U-Net DSC (%) | 3D U-Net HD95 (mm) | Hybrid Swin-UNet DSC (%) | Hybrid Swin-UNet HD95 (mm) | p-value (Wilcoxon) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")

        swin_data = summary.get("hybrid_swin_k5_seed42_ep50", {}).get("per_organ", {})
        unet_data = summary.get("unet3d_k5_seed42_ep50", {}).get("per_organ", {})

        for idx, organ in list(UNIFIED_ORGAN_NAMES.items())[1:]:
            s_dsc = swin_data.get(organ, {}).get("Mean DSC (%)", 0.0)
            s_std = swin_data.get(organ, {}).get("Std DSC (%)", 0.0)
            s_hd = swin_data.get(organ, {}).get("Mean HD95 (mm)", 0.0)

            u_dsc = unet_data.get(organ, {}).get("Mean DSC (%)", 0.0)
            u_std = unet_data.get(organ, {}).get("Std DSC (%)", 0.0)
            u_hd = unet_data.get(organ, {}).get("Mean HD95 (mm)", 0.0)

            p_str = f"{p_vals.get(organ, 1.0):.4f}" if organ in p_vals else "N/A"
            if organ in p_vals and p_vals[organ] < 0.01:
                p_str += " **"
            elif organ in p_vals and p_vals[organ] < 0.05:
                p_str += " *"

            f.write(f"| **{idx}** | **{organ}** | {u_dsc:.1f} ± {u_std:.1f} | {u_hd:.1f} | **{s_dsc:.1f} ± {s_std:.1f}** | **{s_hd:.1f}** | {p_str} |\n")

        f.write("\n\n")

        # Table IV: Few-shot regime summary
        f.write("### TABLE IV: Multi-Regime Few-Shot Generalization Benchmark (k in {1, 3, 5, 10} Shots)\n\n")
        f.write("| Model Architecture | Support Shots (k) | Mean DSC (%) | Mean HD95 (mm) | Mean ASD (mm) | Generalization Gap (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")

        for k in [1, 3, 5, 10]:
            u_key = f"unet3d_k{k}_seed42_ep50"
            s_key = f"hybrid_swin_k{k}_seed42_ep50"

            u_exp = summary.get(u_key, {})
            s_exp = summary.get(s_key, {})

            if u_exp:
                f.write(f"| 3D U-Net Baseline | {k}-shot | {u_exp.get('overall_mean_dsc', 0):.2f}% | {u_exp.get('overall_mean_hd95', 0):.2f} mm | {u_exp.get('overall_mean_asd', 0):.2f} mm | — |\n")
            if s_exp:
                f.write(f"| **Hybrid Swin-UNet (Proposed)** | **{k}-shot** | **{s_exp.get('overall_mean_dsc', 0):.2f}%** | **{s_exp.get('overall_mean_hd95', 0):.2f} mm** | **{s_exp.get('overall_mean_asd', 0):.2f} mm** | **—** |\n")

    print(f"\n>>> [SUCCESS] CBM publication tables generated at {output_md_path}")


def plot_benchmark_figures(summary: Dict[str, Any], curves: Dict[str, Any], output_dir: str = "figures"):
    """Generates high-resolution adaptation and training curves for the paper."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Few-shot adaptation curve
    regimes = [1, 3, 5, 10]
    swin_scores = [summary.get(f"hybrid_swin_k{k}_seed42_ep50", {}).get("overall_mean_dsc", 0.0) for k in regimes]
    unet_scores = [summary.get(f"unet3d_k{k}_seed42_ep50", {}).get("overall_mean_dsc", 0.0) for k in regimes]

    if any(s > 0 for s in swin_scores):
        plt.figure(figsize=(7, 5), dpi=300)
        plt.plot(regimes, swin_scores, marker="o", linewidth=2.5, color="#1f77b4", label="Hybrid Swin-UNet (Proposed)")
        if any(s > 0 for s in unet_scores):
            plt.plot(regimes, unet_scores, marker="s", linewidth=2.0, linestyle="--", color="#d62728", label="3D U-Net Baseline")
        plt.title("Few-Shot Adaptation Performance across Shot Regimes (AMOS -> BTCV)", fontsize=11, fontweight="bold")
        plt.xlabel("Number of Support Volumes (k-shots)", fontsize=10)
        plt.ylabel("Mean Dice Similarity Coefficient (DSC %)", fontsize=10)
        plt.xticks(regimes)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        fig_path = os.path.join(output_dir, "fig4_fewshot_adaptation_curve.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        print(f">>> [FIGURE] Saved few-shot adaptation curve to {fig_path}")

    # 2. Training loss convergence curve
    if curves:
        plt.figure(figsize=(7, 5), dpi=300)
        for exp_key, loss_hist in curves.items():
            if "ep50" in exp_key and len(loss_hist) > 0:
                label = "Hybrid Swin-UNet (5-shot)" if "hybrid_swin" in exp_key else "3D U-Net (5-shot)"
                color = "#1f77b4" if "hybrid_swin" in exp_key else "#d62728"
                plt.plot(range(1, len(loss_hist) + 1), loss_hist, linewidth=1.8, label=label, color=color)
        plt.title("Training Loss Convergence (Dice + Cross-Entropy)", fontsize=11, fontweight="bold")
        plt.xlabel("Adaptation Epochs", fontsize=10)
        plt.ylabel("Loss (L_Dice + L_CE)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        fig_loss_path = os.path.join(output_dir, "fig5_loss_convergence.png")
        plt.tight_layout()
        plt.savefig(fig_loss_path)
        plt.close()
        print(f">>> [FIGURE] Saved loss convergence curve to {fig_loss_path}")


def main():
    summary = load_json("results/benchmark_summary.json")
    per_case = load_json("results/per_case_details.json")
    curves = load_json("results/training_curves.json")

    if not summary:
        print("⚠️ No benchmark_summary.json found in results/ yet. Please run train.py or colab benchmark first.")
        return

    generate_cbm_tables(summary, per_case)
    plot_benchmark_figures(summary, curves)


if __name__ == "__main__":
    main()
