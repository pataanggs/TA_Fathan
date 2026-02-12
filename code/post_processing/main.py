"""
Main pipeline for post-processing ASR predictions.

Reads all_predictions.json saved during training and applies:
  1. Per-fold dictionary building from training transcripts
  2. Text normalization
  3. Dictionary-based spell correction (Levenshtein distance)
  4. WER/CER recomputation at three levels (raw, normalized, corrected)

Results are saved to outputs/<method>/<run_name>/ for each processed run.

Usage:
    cd code/post_processing
    python main.py                          # Latest runs, both methods
    python main.py --method fe              # Freeze Encoder only
    python main.py --method lora            # LoRA only
    python main.py --run run_20260212_060037 --method fe   # Specific FE run
    python main.py --run lora_20260212_063207 --method lora  # Specific LoRA run
    python main.py --list                   # List all available runs
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Set

import numpy as np

from config import (
    CODE_DIR,
    OUTPUT_DIR,
    NUM_FOLDS,
    DICTIONARY_CONFIG,
    discover_runs,
    get_latest_run,
)
from dictionary_builder import (
    build_fold_dictionaries,
)
from evaluator import (
    evaluate_fold,
    compute_cv_summary,
    save_results,
)


# =============================================================================
# Prediction Loading
# =============================================================================

def load_predictions_from_run(run_dir: Path) -> Dict[int, List[Dict[str, str]]]:
    """
    Load all_predictions.json from each fold of a training run.

    Args:
        run_dir: Path to run directory (e.g. .../metrics/run_20260212_060037)

    Returns:
        Dict mapping fold_index → list of {prediction, reference}
    """
    fold_predictions = {}

    for fold in range(NUM_FOLDS):
        pred_file = run_dir / f"fold_{fold}" / "all_predictions.json"
        if not pred_file.exists():
            print(f"  ⚠️  No all_predictions.json for fold {fold}")
            continue

        with open(pred_file, encoding="utf-8") as f:
            predictions = json.load(f)

        fold_predictions[fold] = predictions
        print(f"    Fold {fold}: {len(predictions)} predictions loaded")

    return fold_predictions


# =============================================================================
# Pipeline Execution
# =============================================================================

def process_run(
    run_name: str,
    run_dir: Path,
    method: str,
    method_label: str,
    dictionaries: Dict[int, Dict[str, int]],
) -> Dict[str, Any]:
    """
    Run the full post-processing pipeline for a single training run.

    Args:
        run_name: Name of the run (e.g. 'run_20260212_060037')
        run_dir: Path to the run directory
        method: Method key ('freeze_encoder' or 'lora')
        method_label: Display label ('Freeze Encoder' or 'LoRA')
        dictionaries: Per-fold dictionaries from build_fold_dictionaries

    Returns:
        Cross-validation summary dict
    """
    print(f"\n{'='*70}")
    print(f"  Processing: {method_label}")
    print(f"  Run: {run_name}")
    print(f"{'='*70}")

    # Load predictions
    print(f"\n📂 Loading predictions from {run_name}...")
    fold_predictions = load_predictions_from_run(run_dir)

    if not fold_predictions:
        print(f"  ❌ No predictions found. Did you run training with the updated trainer?")
        print(f"     Expected: {run_dir}/fold_*/all_predictions.json")
        return None

    all_fold_results = []

    for fold in range(NUM_FOLDS):
        if fold not in fold_predictions:
            continue

        print(f"\n{'─'*50}")
        print(f"  Fold {fold}/{NUM_FOLDS-1}")
        print(f"{'─'*50}")

        raw_results = fold_predictions[fold]

        # Evaluate with dictionary
        print(f"  📝 Evaluating with post-processing...")
        dictionary = dictionaries[fold]
        fold_result = evaluate_fold(
            fold=fold,
            raw_results=raw_results,
            dictionary=dictionary,
            max_edit_distance=DICTIONARY_CONFIG["max_edit_distance"],
        )

        # Print fold summary
        raw = fold_result["raw_metrics"]
        norm = fold_result["normalized_metrics"]
        corr = fold_result["corrected_metrics"]
        print(f"\n  📊 Fold {fold} Results:")
        print(f"     {'Level':<20} {'WER%':>8} {'CER%':>8}")
        print(f"     {'─'*38}")
        print(f"     {'Raw':<20} {raw['wer_percent']:>8.2f} {raw['cer_percent']:>8.2f}")
        print(f"     {'Normalized':<20} {norm['wer_percent']:>8.2f} {norm['cer_percent']:>8.2f}")
        print(f"     {'+ Spell Corrected':<20} {corr['wer_percent']:>8.2f} {corr['cer_percent']:>8.2f}")
        print(f"     Corrections applied: {fold_result['num_corrections']}")

        all_fold_results.append(fold_result)

    if not all_fold_results:
        return None

    # Compute cross-validation summary
    cv_summary = compute_cv_summary(all_fold_results)

    # Save results under outputs/<method>/<run_name>
    method_output_dir = OUTPUT_DIR / method
    save_results(
        method_name=run_name,
        fold_results=all_fold_results,
        cv_summary=cv_summary,
        output_dir=method_output_dir,
    )

    return cv_summary


# =============================================================================
# Run Listing
# =============================================================================

def list_available_runs():
    """List all training runs that have all_predictions.json saved."""
    print("\n📋 Available Runs with all_predictions.json:")
    print("=" * 65)

    for method, label in [("freeze_encoder", "Freeze Encoder"), ("lora", "LoRA")]:
        runs = discover_runs(method)
        print(f"\n  {label}:")
        if not runs:
            print(f"    (none found)")
        else:
            for run_name, run_path in runs:
                # Count folds with predictions
                n_folds = sum(
                    1 for f in range(NUM_FOLDS)
                    if (run_path / f"fold_{f}" / "all_predictions.json").exists()
                )
                print(f"    • {run_name}  ({n_folds}/{NUM_FOLDS} folds)")

    print(f"\n  Usage: python main.py --method fe --run <run_name>")
    print(f"         python main.py --method lora --run <run_name>")


# =============================================================================
# Summary Printing
# =============================================================================

def print_summary(results: Dict[str, Dict]):
    """Print a formatted comparison summary."""

    print(f"\n{'='*75}")
    print(f"{'CROSS-VALIDATION SUMMARY — Post-Processing Results':^75}")
    print(f"{'='*75}")

    for label, summary in results.items():
        if summary is None:
            continue

        print(f"\n┌─ {label} {'─'*(70-len(label))}")
        print(f"│  {'Level':<25} {'Mean WER%':>10} {'± Std':>8} {'Mean CER%':>10} {'± Std':>8}")
        print(f"│  {'─'*61}")

        for level_key, level_label in [
            ("raw_metrics", "Raw (model output)"),
            ("normalized_metrics", "Normalized"),
            ("corrected_metrics", "Norm + Spell Corrected"),
        ]:
            s = summary[level_key]
            print(f"│  {level_label:<25} {s['mean_wer']:>10.2f} {s['std_wer']:>7.2f} {s['mean_cer']:>10.2f} {s['std_cer']:>7.2f}")

        imp = summary["improvement"]
        print(f"│")
        print(f"│  📈 Improvement (Raw → Corrected):")
        print(f"│     WER: -{imp['wer_absolute']:.2f}pp ({imp['wer_relative_percent']:.1f}% relative)")
        print(f"│     CER: -{imp['cer_absolute']:.2f}pp ({imp['cer_relative_percent']:.1f}% relative)")
        print(f"│     Total corrections: {summary['total_corrections']}")
        print(f"└{'─'*72}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-processing pipeline for ASR predictions")
    parser.add_argument(
        "--method",
        choices=["fe", "lora", "both"],
        default="both",
        help="Which method to process (default: both)",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specific run name to process (e.g. run_20260212_060037). "
             "If omitted, uses the latest run.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available runs and exit",
    )
    args = parser.parse_args()

    # List mode
    if args.list:
        list_available_runs()
        return

    print("=" * 70)
    print("  ASR Post-Processing Pipeline")
    print("  Dictionary-based Spell Correction with Levenshtein Distance")
    print("=" * 70)

    # Step 1: Build dictionaries
    print("\n📚 Step 1: Building per-fold dictionaries from training transcripts...")
    dictionaries = build_fold_dictionaries()

    for fold, d in dictionaries.items():
        print(f"    Fold {fold}: {len(d)} unique words in dictionary")

    # Step 2: Resolve runs to process
    summaries = {}
    methods_to_process = []

    if args.method in ("fe", "both"):
        methods_to_process.append(("freeze_encoder", "Freeze Encoder"))
    if args.method in ("lora", "both"):
        methods_to_process.append(("lora", "LoRA"))

    for method, label in methods_to_process:
        # Resolve which run to use
        if args.run:
            from config import FE_METRICS_BASE, LORA_METRICS_BASE
            base = FE_METRICS_BASE if method == "freeze_encoder" else LORA_METRICS_BASE
            run_dir = base / args.run
            if not run_dir.exists():
                print(f"\n  ❌ Run '{args.run}' not found at {run_dir}")
                print(f"     Use --list to see available runs.")
                continue
            run_name = args.run
        else:
            run_name, run_dir = get_latest_run(method)
            if run_name is None:
                print(f"\n  ❌ No runs with all_predictions.json found for {label}.")
                print(f"     Run training first, or use --list to check.")
                continue

        summary = process_run(
            run_name=run_name,
            run_dir=run_dir,
            method=method,
            method_label=f"{label} ({run_name})",
            dictionaries=dictionaries,
        )
        summaries[f"{label} ({run_name})"] = summary

    # Step 3: Print final summary
    if summaries:
        print_summary(summaries)

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
