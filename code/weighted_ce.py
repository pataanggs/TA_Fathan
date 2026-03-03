"""
Weighted Cross-Entropy Loss for Whisper Fine-Tuning.

Analyzes previous run predictions to identify frequently mismatched tokens,
then constructs per-token weights for cross-entropy loss. Tokens with higher
error rates receive higher loss weight, forcing the model to pay more
attention to them during training.

Usage:
    from weighted_ce import build_token_weights, WeightedCEMixin

References:
    - Lin et al. (2017). Focal Loss for Dense Object Detection.
    - Szegedy et al. (2016). Rethinking the Inception Architecture (label smoothing).
"""

import json
import math
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

from transformers import WhisperTokenizer


def _discover_prediction_files(
    predictions_dirs: list[str],
    num_folds: int = 5,
) -> list[Path]:
    """
    Auto-discover all_predictions.json files from one or more metrics directories.

    Each entry in predictions_dirs can be:
      - A base metrics dir (e.g., code/lora/outputs/metrics/) → scans all run
        subdirectories inside it.
      - A specific run dir (e.g., .../lora_20260212_063207/) → uses directly.

    Args:
        predictions_dirs: List of directory paths to scan.
        num_folds: Number of CV folds per run.

    Returns:
        List of Path objects pointing to all_predictions.json files found.
    """
    pred_files: list[Path] = []
    seen_runs: set[str] = set()

    for dir_str in predictions_dirs:
        base = Path(dir_str)
        if not base.exists():
            print(f"  ⚠️  Directory not found: {base}")
            continue

        # Check if this is a specific run dir (has fold_0/) or a base metrics dir
        if (base / "fold_0").exists():
            run_dirs = [base]
        else:
            # Base metrics dir — discover all run subdirectories
            run_dirs = sorted(
                [d for d in base.iterdir() if d.is_dir() and (d / "fold_0").exists()]
            )

        for run_dir in run_dirs:
            if run_dir.name in seen_runs:
                continue
            seen_runs.add(run_dir.name)

            for fold in range(num_folds):
                pf = run_dir / f"fold_{fold}" / "all_predictions.json"
                if pf.exists():
                    pred_files.append(pf)

    return pred_files


def analyze_token_errors(
    predictions_dirs: list[str],
    tokenizer: WhisperTokenizer,
    num_folds: int = 5,
) -> Tuple[Counter, Counter]:
    """
    Analyze all_predictions.json from ALL available runs to find token-level errors.

    Aggregates error statistics across every run and fold found in
    predictions_dirs, giving a more robust signal than a single run.

    For each mismatched sample, compares token bags (reference vs prediction)
    to count how often each token is involved in errors.

    Args:
        predictions_dirs: List of metrics directory paths (base dirs or specific runs).
        tokenizer: WhisperTokenizer for encoding text → token IDs
        num_folds: Number of CV folds per run

    Returns:
        (error_counts, ref_counts): Counters of per-token error and reference frequencies
    """
    pred_files = _discover_prediction_files(predictions_dirs, num_folds)

    if not pred_files:
        print("  ⚠️  No prediction files found in any of the provided directories")
        return Counter(), Counter()

    # Count unique runs for reporting
    run_names = set(pf.parent.parent.name for pf in pred_files)
    print(f"   📂 Found {len(pred_files)} prediction files across {len(run_names)} runs")
    for rn in sorted(run_names):
        print(f"      • {rn}")

    error_counts = Counter()
    ref_counts = Counter()

    for pred_file in pred_files:
        with open(pred_file, encoding="utf-8") as f:
            preds = json.load(f)

        for sample in preds:
            ref = sample["reference"]
            hyp = sample["prediction"]

            ref_ids = tokenizer.encode(ref, add_special_tokens=False)
            hyp_ids = tokenizer.encode(hyp, add_special_tokens=False)

            # Count reference token frequencies
            for tid in ref_ids:
                ref_counts[tid] += 1

            if ref != hyp:
                ref_bag = Counter(ref_ids)
                hyp_bag = Counter(hyp_ids)

                # Missed tokens (in ref but not/less in hyp)
                for tid, count in ref_bag.items():
                    diff = count - hyp_bag.get(tid, 0)
                    if diff > 0:
                        error_counts[tid] += diff

                # Hallucinated tokens (in hyp but not/less in ref)
                for tid, count in hyp_bag.items():
                    diff = count - ref_bag.get(tid, 0)
                    if diff > 0:
                        error_counts[tid] += diff

    return error_counts, ref_counts


def build_token_weights(
    predictions_dirs: list[str],
    tokenizer: WhisperTokenizer,
    num_folds: int = 5,
    base_weight: float = 1.0,
    max_weight: float = 3.0,
    min_error_count: int = 3,
    smoothing: float = 0.5,
    model_vocab_size: int | None = None,
) -> torch.Tensor:
    """
    Build a per-token weight vector from previous prediction errors.

    Aggregates token error statistics across ALL runs found in the provided
    directories, producing more robust weights than a single run.

    Weight formula for error-prone tokens:
        weight = base_weight + (max_weight - base_weight) * sigmoid(error_rate - 0.5)

    where error_rate = error_count / ref_count (clipped to [0, 1]).

    Tokens with < min_error_count errors keep base_weight to avoid
    noise from rare tokens.

    Args:
        predictions_dirs: List of metrics directory paths. Each can be a base
            metrics dir (auto-discovers all runs) or a specific run dir.
        tokenizer: WhisperTokenizer
        num_folds: Number of CV folds per run
        base_weight: Default weight for all tokens (1.0 = standard CE)
        max_weight: Maximum weight for the most error-prone tokens
        min_error_count: Minimum error occurrences to apply boosted weight
        smoothing: Sigmoid steepness control (lower = smoother transition)
        model_vocab_size: Override vocab size (use model.config.vocab_size to
            match the model's output dim, which may be larger than
            tokenizer.vocab_size due to special tokens). If None, uses
            tokenizer.vocab_size.

    Returns:
        torch.Tensor of shape (vocab_size,) with per-token weights
    """
    vocab_size = model_vocab_size if model_vocab_size is not None else tokenizer.vocab_size
    weights = torch.full((vocab_size,), base_weight, dtype=torch.float32)

    error_counts, ref_counts = analyze_token_errors(
        predictions_dirs, tokenizer, num_folds
    )

    boosted = 0
    for tid, err_count in error_counts.items():
        if tid >= vocab_size:
            continue
        if err_count < min_error_count:
            continue

        ref_count = max(ref_counts.get(tid, 1), 1)
        error_rate = min(err_count / ref_count, 1.0)

        # Sigmoid scaling: smooth transition from base_weight to max_weight
        # centered at error_rate = 0.3 (tokens with 30%+ error rate get boosted)
        x = (error_rate - 0.3) / smoothing
        sigmoid = 1.0 / (1.0 + math.exp(-x))
        weight = base_weight + (max_weight - base_weight) * sigmoid

        weights[tid] = weight
        boosted += 1

    print(f"\n⚖️  Weighted Cross-Entropy Token Analysis:")
    print(f"   Vocab size:        {vocab_size:,}")
    print(f"   Unique ref tokens: {len(ref_counts)}")
    print(f"   Error-prone tokens (>= {min_error_count} errors): {len([t for t, c in error_counts.items() if c >= min_error_count])}")
    print(f"   Boosted tokens:    {boosted}")
    print(f"   Weight range:      [{base_weight:.2f}, {max_weight:.2f}]")
    print(f"\n   Top 15 highest-weighted tokens:")
    top_tokens = sorted(
        [(tid, weights[tid].item()) for tid in range(vocab_size) if weights[tid] > base_weight],
        key=lambda x: x[1],
        reverse=True,
    )
    for tid, w in top_tokens[:15]:
        token_str = tokenizer.decode([tid]).strip()
        err = error_counts.get(tid, 0)
        ref = ref_counts.get(tid, 0)
        print(f"   Token {tid:5d}: '{token_str:15s}' | weight={w:.3f} | errors={err:3d}/{ref:3d} ({100*err/max(ref,1):.0f}%)")

    return weights


class WeightedCEMixin:
    """
    Mixin that overrides compute_loss() to use weighted cross-entropy.

    Supports label_smoothing_factor from training args.
    Token weights are applied via the `weight` parameter of F.cross_entropy,
    which scales the loss contribution of each target token proportionally.

    Usage:
        class MyTrainer(WeightedCEMixin, Seq2SeqTrainer):
            pass

        trainer = MyTrainer(token_weights=weights_tensor, ...)
    """

    def __init__(self, *args, token_weights: Optional[torch.Tensor] = None, **kwargs):
        # Extract token_weights before passing to parent (parent doesn't expect it)
        self._token_weights = token_weights
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        """
        Compute weighted cross-entropy loss with optional label smoothing.

        1. Pre-compute decoder_input_ids (Whisper label-smoothing fix)
        2. Forward pass to get logits
        3. Apply weighted CE with label smoothing
        """
        # --- Whisper label-smoothing fix: inject decoder_input_ids ---
        if "labels" in inputs and "decoder_input_ids" not in inputs:
            labels = inputs["labels"]
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = model.config.decoder_start_token_id
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == -100,
                model.config.pad_token_id,
            )
            inputs["decoder_input_ids"] = decoder_input_ids

        # If no token weights, fall back to parent (standard CE)
        if self._token_weights is None:
            return super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)

        # --- Weighted CE path ---
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Move weights to same device as logits, pad if vocab size changed
        weights = self._token_weights.to(logits.device)
        model_vocab = logits.size(-1)
        if weights.size(0) < model_vocab:
            pad = torch.full(
                (model_vocab - weights.size(0),),
                weights[0].item(),  # use base_weight
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = torch.cat([weights, pad])

        # Label smoothing factor from training args
        ls = self.args.label_smoothing_factor if self.args.label_smoothing_factor else 0.0

        if ls > 0.0:
            # Manual label-smoothed weighted CE
            # Standard label smoothing distributes `ls` probability uniformly
            # and keeps `(1 - ls)` on the target token.
            vocab_size = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

            # One-hot with smoothing
            # smooth_targets[i] = ls / V for all i, then += (1 - ls) for target
            nll_loss = F.nll_loss(
                log_probs.view(-1, vocab_size),
                labels.view(-1),
                weight=weights,
                ignore_index=-100,
                reduction="sum",
            )

            # Smoothing component: average log_prob weighted by token weights
            # For each position, smooth_loss = -mean(log_probs) weighted by target token weight
            pad_mask = labels.view(-1) != -100
            if pad_mask.any():
                # Get weight for each target token
                valid_labels = labels.view(-1).clamp(min=0)  # replace -100 with 0 for indexing
                token_w = weights[valid_labels] * pad_mask.float()  # zero out padding

                # Weighted smooth loss
                smooth_loss = -(log_probs.view(-1, vocab_size).sum(dim=-1) * token_w).sum() / vocab_size
                num_active = token_w.sum().clamp(min=1.0)

                loss = ((1.0 - ls) * nll_loss + ls * smooth_loss) / num_active
            else:
                loss = nll_loss
        else:
            # Standard weighted CE without label smoothing
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=weights,
                ignore_index=-100,
            )

        return loss


if __name__ == "__main__":
    """Quick test: analyze token errors and build weights from ALL runs."""
    tokenizer = WhisperTokenizer.from_pretrained(
        "openai/whisper-base", language="indonesian", task="transcribe"
    )

    # Scan both strategy metrics directories — aggregates ALL runs
    all_dirs = [
        "code/lora/outputs/metrics",
        "code/freeze_encoder/outputs/metrics",
    ]
    existing = [d for d in all_dirs if Path(d).exists()]

    if existing:
        print(f"\n{'='*60}")
        print(f"Building weights from ALL runs in {len(existing)} directories")
        print(f"{'='*60}")
        weights = build_token_weights(existing, tokenizer)
        print(f"\nWeight tensor shape: {weights.shape}")
        print(f"Non-default weights: {(weights != 1.0).sum().item()}")
    else:
        print("No metrics directories found.")
