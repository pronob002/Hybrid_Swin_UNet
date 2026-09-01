"""
train.py
========
Top-level entrypoint for Few-Shot Cross-Domain 3D Abdominal Multi-Organ Segmentation.
"""

from scripts.train_fewshot import run_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot 3D Multi-Organ Segmentation Benchmark")
    parser.add_argument("--model", type=str, default="hybrid_swin", choices=["hybrid_swin", "unet3d"],
                        help="Model architecture to train (default: hybrid_swin)")
    parser.add_argument("--shots", type=int, default=2,
                        help="Number of support volumes from AMOS (default: 2)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of adaptation epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--eval_cases", type=int, default=2,
                        help="Number of target BTCV evaluation cases (default: 2)")
    args = parser.parse_args()

    run_experiment(
        model_type=args.model,
        k_shots=args.shots,
        num_epochs=args.epochs,
        seed=args.seed,
        eval_cases=args.eval_cases
    )
