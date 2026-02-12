"""
Evaluator module for post-processing.
Applies text normalization and spell correction, then re-computes WER/CER.

Produces three metric levels for comparison:
  1. Raw predictions (as-is from model)
  2. Normalized predictions + references (fair text normalization)
  3. Normalized + corrected predictions vs normalized references (spell correction)

Note: Model loading and inference functions are kept as utilities but the
primary workflow now loads predictions from all_predictions.json files
saved during training.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Union

import torch
import evaluate

from config import (
    CODE_DIR,
    OUTPUT_DIR,
    NUM_FOLDS,
)
from text_normalizer import normalize_text, normalize_pair
from spell_corrector import correct_sentence


# =============================================================================
# Model Loading
# =============================================================================

def load_fe_model_and_processor(checkpoint_dir: str):
    """
    Load a Freeze Encoder checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory containing model.safetensors

    Returns:
        (model, processor) tuple
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)

    # Load processor from the same checkpoint (it saves tokenizer files)
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)

    model.eval()
    return model, processor


def load_lora_model_and_processor(checkpoint_dir: str):
    """
    Load a LoRA checkpoint (adapter weights on base model).

    Args:
        checkpoint_dir: Path to checkpoint directory containing adapter_model.safetensors

    Returns:
        (model, processor) tuple
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    # Load base model
    base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # Load LoRA adapter from checkpoint
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()  # Merge for faster inference

    # Load processor from the checkpoint
    processor = WhisperProcessor.from_pretrained(checkpoint_dir)

    model.eval()
    return model, processor


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model,
    processor,
    eval_dataset,
    device: str = "cuda",
) -> List[Dict[str, str]]:
    """
    Run inference on the full evaluation dataset.

    Args:
        model: Loaded Whisper model
        processor: Whisper processor/tokenizer
        eval_dataset: HuggingFace Dataset with input_features and labels
        device: Device to run on

    Returns:
        List of dicts: [{prediction, reference}, ...]
    """
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

        # Decode reference
        labels = sample["labels"]
        labels = [l if l != -100 else processor.tokenizer.pad_token_id for l in labels]
        ref_text = processor.tokenizer.decode(labels, skip_special_tokens=True)

        results.append({
            "prediction": pred_text,
            "reference": ref_text,
        })

    return results


# =============================================================================
# Metric Computation
# =============================================================================

def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute WER and CER for a list of predictions and references.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dict with wer, cer, wer_percent, cer_percent
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Filter out empty pairs
    valid_preds = []
    valid_refs = []
    for p, r in zip(predictions, references):
        if p.strip() and r.strip():
            valid_preds.append(p)
            valid_refs.append(r)

    if not valid_preds:
        return {"wer": 1.0, "cer": 1.0, "wer_percent": 100.0, "cer_percent": 100.0}

    wer = wer_metric.compute(predictions=valid_preds, references=valid_refs)
    cer = cer_metric.compute(predictions=valid_preds, references=valid_refs)

    return {
        "wer": wer,
        "cer": cer,
        "wer_percent": round(wer * 100, 4),
        "cer_percent": round(cer * 100, 4),
    }


def compute_per_sample_metrics(
    predictions: List[str],
    references: List[str],
) -> List[Dict[str, float]]:
    """
    Compute per-sample WER and CER.
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    results = []

    for pred, ref in zip(predictions, references):
        if pred.strip() and ref.strip():
            w = wer_metric.compute(predictions=[pred], references=[ref])
            c = cer_metric.compute(predictions=[pred], references=[ref])
        else:
            w, c = 1.0, 1.0
        results.append({"wer": w, "cer": c})

    return results


# =============================================================================
# Full Fold Evaluation
# =============================================================================

def evaluate_fold(
    fold: int,
    raw_results: List[Dict[str, str]],
    dictionary: Union[Set[str], Dict[str, int]],
    max_edit_distance: int = 2,
) -> Dict[str, Any]:
    """
    Evaluate a single fold with three metric levels:
      1. Raw (as-is from model)
      2. Normalized (both pred + ref normalized)
      3. Normalized + corrected (spell correction on pred only)

    Args:
        fold: Fold index
        raw_results: List of {prediction, reference} from inference
        dictionary: Set of valid words for this fold
        max_edit_distance: Max Levenshtein distance for correction

    Returns:
        Comprehensive evaluation results dict
    """
    raw_preds = [r["prediction"] for r in raw_results]
    raw_refs = [r["reference"] for r in raw_results]

    # Level 1: Raw metrics
    raw_metrics = compute_metrics(raw_preds, raw_refs)

    # Level 2: Normalized
    norm_pairs = [normalize_pair(p, r) for p, r in zip(raw_preds, raw_refs)]
    norm_preds = [p for p, r in norm_pairs]
    norm_refs = [r for p, r in norm_pairs]
    norm_metrics = compute_metrics(norm_preds, norm_refs)

    # Level 3: Normalized + Spell Corrected
    corrected_preds = []
    all_corrections = []
    for pred in norm_preds:
        corrected, corrections = correct_sentence(pred, dictionary, max_edit_distance)
        corrected_preds.append(corrected)
        all_corrections.extend(corrections)

    corrected_metrics = compute_metrics(corrected_preds, norm_refs)

    # Per-sample details (corrected level)
    per_sample = []
    per_sample_metrics = compute_per_sample_metrics(corrected_preds, norm_refs)
    for i, (raw_p, raw_r, norm_p, norm_r, corr_p, m) in enumerate(
        zip(raw_preds, raw_refs, norm_preds, norm_refs, corrected_preds, per_sample_metrics)
    ):
        entry = {
            "index": i,
            "raw_prediction": raw_p,
            "raw_reference": raw_r,
            "normalized_prediction": norm_p,
            "normalized_reference": norm_r,
            "corrected_prediction": corr_p,
            "wer": m["wer"],
            "cer": m["cer"],
        }
        # Track corrections for this sample
        _, sample_corrections = correct_sentence(norm_p, dictionary, max_edit_distance)
        if sample_corrections:
            entry["corrections"] = sample_corrections
        per_sample.append(entry)

    return {
        "fold": fold,
        "num_samples": len(raw_results),
        "raw_metrics": raw_metrics,
        "normalized_metrics": norm_metrics,
        "corrected_metrics": corrected_metrics,
        "num_corrections": len(all_corrections),
        "unique_corrections": len(set((c["original"], c["corrected"]) for c in all_corrections)),
        "sample_corrections": all_corrections[:50],  # Limit for readability
        "per_sample_results": per_sample,
    }


# =============================================================================
# Cross-Validation Summary
# =============================================================================

def compute_cv_summary(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute cross-validation summary statistics across all folds.
    """
    import numpy as np

    summary = {}

    for level in ["raw_metrics", "normalized_metrics", "corrected_metrics"]:
        wers = [f[level]["wer_percent"] for f in fold_results]
        cers = [f[level]["cer_percent"] for f in fold_results]

        summary[level] = {
            "mean_wer": round(float(np.mean(wers)), 4),
            "std_wer": round(float(np.std(wers)), 4),
            "min_wer": round(float(np.min(wers)), 4),
            "max_wer": round(float(np.max(wers)), 4),
            "mean_cer": round(float(np.mean(cers)), 4),
            "std_cer": round(float(np.std(cers)), 4),
            "min_cer": round(float(np.min(cers)), 4),
            "max_cer": round(float(np.max(cers)), 4),
            "per_fold_wer": wers,
            "per_fold_cer": cers,
        }

    # Improvement from raw → corrected
    raw_wer = summary["raw_metrics"]["mean_wer"]
    corr_wer = summary["corrected_metrics"]["mean_wer"]
    raw_cer = summary["raw_metrics"]["mean_cer"]
    corr_cer = summary["corrected_metrics"]["mean_cer"]

    summary["improvement"] = {
        "wer_absolute": round(raw_wer - corr_wer, 4),
        "cer_absolute": round(raw_cer - corr_cer, 4),
        "wer_relative_percent": round((raw_wer - corr_wer) / raw_wer * 100, 2) if raw_wer > 0 else 0,
        "cer_relative_percent": round((raw_cer - corr_cer) / raw_cer * 100, 2) if raw_cer > 0 else 0,
    }

    # Total corrections
    total_corrections = sum(f["num_corrections"] for f in fold_results)
    total_unique = sum(f["unique_corrections"] for f in fold_results)
    summary["total_corrections"] = total_corrections
    summary["total_unique_corrections"] = total_unique

    return summary


def save_results(
    method_name: str,
    fold_results: List[Dict[str, Any]],
    cv_summary: Dict[str, Any],
    output_dir: Path,
):
    """
    Save all evaluation results to disk.

    Creates:
      - output_dir/<method>/fold_<i>/evaluation_results.json
      - output_dir/<method>/cross_validation_summary.json
    """
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    # Save per-fold results
    for fold_result in fold_results:
        fold_dir = method_dir / f"fold_{fold_result['fold']}"
        fold_dir.mkdir(exist_ok=True)

        with open(fold_dir / "evaluation_results.json", "w") as f:
            json.dump(fold_result, f, indent=2, ensure_ascii=False)

    # Save cross-validation summary
    with open(method_dir / "cross_validation_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    print(f"  💾 Results saved to {method_dir}")


if __name__ == "__main__":
    print("Evaluator module — run via main.py for full pipeline")
