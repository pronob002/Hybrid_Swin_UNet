"""
scripts/cleanup_checkpoints.py
==============================
Storage Optimizer for Hugging Face Hub & Local Disk.

Scans your Hugging Face cloud repository and local checkpoints directory,
identifies redundant intermediate files (e.g. checkpoint_latest.pth when
checkpoint_best.pth is available), and safely purges them to prevent free-tier
storage quota exhaustion.

Can be run standalone via CLI or imported into notebooks.
"""

import os
import sys
import argparse
from typing import List, Dict, Optional, Any
from huggingface_hub import HfApi


def cleanup_redundant_checkpoints(
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    delete_local: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Identifies and deletes redundant checkpoint_latest.pth files where
    checkpoint_best.pth is present, freeing cloud and local storage.
    """
    token = token or os.environ.get("HF_TOKEN")
    results = {
        "remote_deleted": [],
        "remote_kept": [],
        "local_deleted": [],
        "local_kept": [],
        "reclaimed_bytes_local": 0,
        "estimated_reclaimed_bytes_remote": 0,
    }

    print("=" * 80)
    print(" [STORAGE CLEANUP] >>> HUGGING FACE CLOUD & LOCAL STORAGE OPTIMIZER")
    if repo_id:
        print(f" Target Cloud Repo : https://huggingface.co/{repo_id}")
    print(f" Local Checkpoint  : {checkpoint_dir}")
    print(f" Execution Mode    : {'DRY RUN (preview only)' if dry_run else 'ACTIVE (purge files)'}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Cloud Cleanup on Hugging Face Hub
    # -------------------------------------------------------------------------
    if repo_id and token:
        try:
            api = HfApi(token=token)
            remote_files = api.list_repo_files(repo_id=repo_id, repo_type="model", token=token)
            
            # Find all experiments with checkpoints
            exp_folders = set()
            for f in remote_files:
                if f.startswith("checkpoints/") and ("checkpoint_best.pth" in f or "checkpoint_latest.pth" in f):
                    parts = f.split("/")
                    if len(parts) >= 3:
                        exp_folders.add("/".join(parts[:-1]))

            remote_to_delete = []
            for folder in sorted(exp_folders):
                best_file = f"{folder}/checkpoint_best.pth"
                latest_file = f"{folder}/checkpoint_latest.pth"

                if best_file in remote_files and latest_file in remote_files:
                    remote_to_delete.append(latest_file)
                    results["remote_kept"].append(best_file)
                elif latest_file in remote_files:
                    # Only latest exists, keep it so work isn't lost
                    results["remote_kept"].append(latest_file)

            results["remote_deleted"] = remote_to_delete
            # Estimated 33.8MB per Hybrid Swin / UNet3D checkpoint
            results["estimated_reclaimed_bytes_remote"] = len(remote_to_delete) * 33_800_000

            if remote_to_delete:
                print(f"\n>>> Found {len(remote_to_delete)} redundant 'checkpoint_latest.pth' file(s) on Hugging Face Hub:")
                for rf in remote_to_delete:
                    print(f"  [-] To Delete: {rf}")
                
                if not dry_run:
                    api.delete_files(
                        repo_id=repo_id,
                        delete_patterns=remote_to_delete,
                        repo_type="model",
                        token=token,
                        commit_message="Storage Cleanup: Purge redundant intermediate checkpoints"
                    )
                    print(f">>> Successfully deleted {len(remote_to_delete)} redundant checkpoint(s) from Hugging Face Hub!")
            else:
                print("\n>>> No redundant checkpoints found on Hugging Face Hub (Cloud storage is clean).")

        except Exception as e:
            print(f"[WARNING] Cloud cleanup encountered an error: {e}")
    else:
        print("\n>>> Hugging Face Hub credentials not provided or incomplete. Skipping cloud cleanup.")

    # -------------------------------------------------------------------------
    # 2. Local Disk Cleanup
    # -------------------------------------------------------------------------
    if delete_local and os.path.isdir(checkpoint_dir):
        local_to_delete = []
        for root, dirs, files in os.walk(checkpoint_dir):
            if "checkpoint_best.pth" in files and "checkpoint_latest.pth" in files:
                latest_path = os.path.join(root, "checkpoint_latest.pth")
                local_to_delete.append(latest_path)
                results["local_kept"].append(os.path.join(root, "checkpoint_best.pth"))
            elif "checkpoint_best.pth" in files:
                results["local_kept"].append(os.path.join(root, "checkpoint_best.pth"))
            elif "checkpoint_latest.pth" in files:
                results["local_kept"].append(os.path.join(root, "checkpoint_latest.pth"))

        for lp in local_to_delete:
            try:
                size = os.path.getsize(lp)
                if not dry_run:
                    os.remove(lp)
                results["local_deleted"].append(lp)
                results["reclaimed_bytes_local"] += size
            except Exception as e:
                print(f"[WARNING] Could not delete local file {lp}: {e}")

        if local_to_delete:
            print(f"\n>>> Found {len(local_to_delete)} redundant local checkpoint(s):")
            for lp in local_to_delete:
                print(f"  [-] {'Deleted' if not dry_run else 'To Delete'}: {lp}")
        else:
            print("\n>>> Local checkpoint directory is already optimized.")

    # -------------------------------------------------------------------------
    # 3. Summary Report
    # -------------------------------------------------------------------------
    reclaimed_mb_cloud = results["estimated_reclaimed_bytes_remote"] / (1024 * 1024)
    reclaimed_mb_local = results["reclaimed_bytes_local"] / (1024 * 1024)

    print("\n" + "=" * 80)
    print(" [STORAGE CLEANUP SUMMARY]")
    print(f"  Remote Checkpoints Deleted : {len(results['remote_deleted'])}")
    print(f"  Remote Checkpoints Retained: {len(results['remote_kept'])} (All optimal weights preserved)")
    print(f"  Estimated Cloud Space Saved: ~{reclaimed_mb_cloud:.1f} MB")
    print(f"  Local Checkpoints Deleted  : {len(results['local_deleted'])}")
    print(f"  Local Disk Space Saved     : {reclaimed_mb_local:.1f} MB")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face & Local Checkpoint Storage Cleanup")
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face model repository ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Local checkpoint directory")
    parser.add_argument("--no_local", action="store_true", help="Skip local disk cleanup")
    parser.add_argument("--dry_run", action="store_true", help="Show files to be deleted without deleting")
    args = parser.parse_args()

    cleanup_redundant_checkpoints(
        repo_id=args.hf_repo,
        token=args.hf_token,
        checkpoint_dir=args.checkpoint_dir,
        delete_local=not args.no_local,
        dry_run=args.dry_run
    )
