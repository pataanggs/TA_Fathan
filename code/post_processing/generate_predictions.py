"""
One-time migration script: Generate all_predictions.json for existing runs.

Since save_all_predictions() was added AFTER the latest training runs,
this script runs inference from existing checkpoints and saves predictions
to the correct metrics directories so discover_runs() can find them.

Usage:
    cd code/post_processing
    python generate_predictions.py --method fe    # For Freeze Encoder
    python generate_predictions.py --method lora  # For LoRA
    python generate_predictions.py                # Both
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import KFold

from config import (
    CODE_DIR,
    DATA_DIR,
    METADATA_CSV,
    CSV_COLUMNS,
    NUM_FOLDS,
    RANDOM_STATE,
    FE_METRICS_BASE,
    LORA_METRICS_BASE,
)


def find_checkpoints(method: str):
    """Find the best checkpoint for each fold."""
    checkpoints_base = CODE_DIR / method / "outputs" / "checkpoints"
    fold_checkpoints = {}

    for fold in range(NUM_FOLDS):
        fold_dir = checkpoints_base / f"fold_{fold}"
        if not fold_dir.exists():
            continue
        cp_dirs = sorted(fold_dir.glob("checkpoint-*"))
        if cp_dirs:
            fold_checkpoints[fold] = cp_dirs[-1]
    return fold_checkpoints


def find_latest_run(method: str) -> str:
    """Find the latest metrics run directory."""
    base = FE_METRICS_BASE if method == "freeze_encoder" else LORA_METRICS_BASE
    if not base.exists():
        return None
    runs = sorted([d for d in base.iterdir() if d.is_dir()], reverse=True)
    return runs[0].name if runs else None


def run_inference_for_fold(model, processor, eval_dataset, device):
    """Run inference on all eval samples."""
    model = model.to(device)
    results = []

    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                attention_mask=attention_mask,
            )

        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        labels = sample["labels"]
        labels = [l if l != -100 else processor.tokenizer.pad_token_id for l in labels]
        ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)

        results.append({
            "prediction": pred_text,
            "reference": ref_text,
        })

    return results


def build_eval_datasets(processor, method: str):
    """Build eval datasets for each fold using the same KFold split as training."""
    import importlib

    method_dir = str(CODE_DIR / method)
    code_dir = str(CODE_DIR)

    saved_path = sys.path.copy()
    sys.path = [method_dir, code_dir] + [p for p in sys.path if p not in (method_dir, code_dir)]

    for mod_name in ["config", "data_loader", "dataset", "audio_preprocessing", "text_preprocessing", "augmentation"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    import data_loader
    import dataset as ds_module
    importlib.reload(data_loader)
    importlib.reload(ds_module)

    from data_loader import load_minang_wav_metadata, dataframe_to_hf_dataset, get_train_eval_split
    from dataset import prepare_dataset_for_evaluation

    df = load_minang_wav_metadata()
    if df is None:
        df = pd.read_csv(METADATA_CSV, header=None, names=CSV_COLUMNS)
        df["full_path"] = df["wav_path"].apply(lambda x: str(DATA_DIR / x))

    hf_dataset = dataframe_to_hf_dataset(df)

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    indices = list(range(len(hf_dataset)))

    eval_datasets = {}
    for fold, (train_idx, eval_idx) in enumerate(kfold.split(indices)):
        _, eval_ds = get_train_eval_split(hf_dataset, train_idx.tolist(), eval_idx.tolist())
        eval_ds = prepare_dataset_for_evaluation(eval_ds, processor)
        eval_datasets[fold] = eval_ds
        print(f"    Fold {fold}: {len(eval_ds)} eval samples")

    sys.path = saved_path
    return eval_datasets


def generate_for_method(method: str, method_label: str):
    """Generate all_predictions.json for each fold of a method."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print(f"\n{'='*70}")
    print(f"  Generating predictions for: {method_label}")
    print(f"{'='*70}")

    # Find checkpoints
    checkpoints = find_checkpoints(method)
    if not checkpoints:
        print(f"  ❌ No checkpoints found for {method}")
        return

    # Find latest run name
    run_name = find_latest_run(method)
    if not run_name:
        print(f"  ❌ No metrics runs found for {method}")
        return

    metrics_base = FE_METRICS_BASE if method == "freeze_encoder" else LORA_METRICS_BASE
    run_dir = metrics_base / run_name

    print(f"  📂 Checkpoints found for {len(checkpoints)} folds")
    print(f"  📁 Target run: {run_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  🖥️  Device: {device}")

    for fold in range(NUM_FOLDS):
        if fold not in checkpoints:
            print(f"\n  ⚠️  Fold {fold}: no checkpoint, skipping")
            continue

        save_path = run_dir / f"fold_{fold}" / "all_predictions.json"
        if save_path.exists():
            print(f"\n  ✅ Fold {fold}: all_predictions.json already exists, skipping")
            continue

        print(f"\n  {'─'*50}")
        print(f"  Fold {fold}/{NUM_FOLDS-1}")
        print(f"  {'─'*50}")

        cp_path = str(checkpoints[fold])
        print(f"  🔄 Loading model from {checkpoints[fold].name}...")
        t0 = time.time()

        if method == "freeze_encoder":
            model = WhisperForConditionalGeneration.from_pretrained(cp_path)
            processor = WhisperProcessor.from_pretrained(cp_path)
        else:
            from peft import PeftModel
            base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
            model = PeftModel.from_pretrained(base_model, cp_path)
            model = model.merge_and_unload()
            processor = WhisperProcessor.from_pretrained(cp_path)

        model.eval()
        print(f"     Model loaded in {time.time()-t0:.1f}s")

        # Build eval dataset
        print(f"  📊 Building eval dataset...")
        eval_datasets = build_eval_datasets(processor, method)
        eval_ds = eval_datasets[fold]

        # Run inference
        print(f"  🎤 Running inference on {len(eval_ds)} samples...")
        t0 = time.time()
        results = run_inference_for_fold(model, processor, eval_ds, device)
        print(f"     Inference completed in {time.time()-t0:.1f}s")

        # Save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved {len(results)} predictions to {save_path}")

        # Free memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n  ✅ Done generating predictions for {method_label}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all_predictions.json from existing checkpoints"
    )
    parser.add_argument(
        "--method",
        choices=["fe", "lora", "both"],
        default="both",
        help="Which method to process (default: both)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  One-time Migration: Generate all_predictions.json")
    print("  from existing checkpoints → metrics directories")
    print("=" * 70)

    if args.method in ("fe", "both"):
        generate_for_method("freeze_encoder", "Freeze Encoder")

    if args.method in ("lora", "both"):
        generate_for_method("lora", "LoRA")

    print("\n🎉 Migration complete! Now run:")
    print("   python main.py --list")
    print("   python main.py")


if __name__ == "__main__":
    main()
