"""
train.py
========
Top-level CLI execution entrypoint for Few-Shot Cross-Domain 3D Abdominal Multi-Organ Segmentation.
Includes automated checkpointing, seamless resume logic, persistent result caching,
and optional Hugging Face Hub cloud auto-sync.
"""

from scripts.train_fewshot import run_experiment
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-Shot 3D Multi-Organ Segmentation Benchmark")
    parser.add_argument("--model", type=str, default="hybrid_swin", choices=["hybrid_swin", "unet3d"],
                        help="Model architecture to train (default: hybrid_swin)")
    parser.add_argument("--shots", type=int, default=5,
                        help="Number of support volumes from AMOS (default: 5)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of adaptation epochs (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--eval_cases", type=int, default=10,
                        help="Number of target BTCV evaluation cases (default: 10)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save/load model checkpoints (default: checkpoints)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save persistent summary CSV & JSON (default: results)")
    parser.add_argument("--hf_repo", type=str, default=None,
                        help="Hugging Face Model Repo ID (e.g. 'username/hybrid-swin-unet-checkpoints')")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face Access Token (or set HF_TOKEN env var)")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Force rerun even if experiment already exists in results cache")
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
