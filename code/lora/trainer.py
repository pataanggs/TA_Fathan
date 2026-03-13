"""
Trainer module for Whisper LoRA fine-tuning.
Implements model initialization with PEFT LoRA adapter and training loop.
Uses Parameter-Efficient Fine-Tuning to train only ~0.27% of parameters.
"""

import torch
import evaluate
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)

from config import (
    MODEL_NAME,
    LANGUAGE,
    LANGUAGE_FULL,
    TASK,
    TRAINING_ARGS,
    MODEL_DROPOUT_CONFIG,
    GENERATION_CONFIG,
    LORA_CONFIG,
    EARLY_STOPPING_CONFIG,
    METRICS_LOGGING_CONFIG,
    WEIGHTED_CE_CONFIG,
    OUTPUT_DIR,
)

from metrics_logger import (
    MetricsLogger,
    create_metrics_logger,
    create_metrics_callbacks,
    ComprehensiveMetricsCallback,
    PredictionLoggingCallback,
)

# Add parent dir (code/) to path for shared modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from weighted_ce import WeightedCEMixin, build_token_weights


class WhisperLoRATrainer(WeightedCEMixin, Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer for Whisper + LoRA.
    
    Inherits WeightedCEMixin which handles:
    1. Pre-computing decoder_input_ids for label smoothing compatibility
    2. Weighted cross-entropy loss when token_weights is provided
    3. Falls back to standard CE when token_weights is None
    """
    pass


def load_model() -> WhisperForConditionalGeneration:
    """
    Load and initialize Whisper model with proper language configuration.

    Returns:
        WhisperForConditionalGeneration model configured for Indonesian
    """
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Configure generation config with Indonesian language
    model.generation_config.language = LANGUAGE  # "id"
    model.generation_config.task = TASK  # "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Configure beam search for inference
    model.generation_config.num_beams = GENERATION_CONFIG["num_beams"]
    model.generation_config.max_length = GENERATION_CONFIG["max_length"]

    # Apply dropout configuration
    model.config.dropout = MODEL_DROPOUT_CONFIG["dropout"]
    model.config.attention_dropout = MODEL_DROPOUT_CONFIG["attention_dropout"]
    model.config.activation_dropout = MODEL_DROPOUT_CONFIG["activation_dropout"]

    return model


def apply_lora(model: WhisperForConditionalGeneration) -> PeftModel:
    """
    Apply LoRA adapters to the Whisper model.
    Only ~0.27% of parameters become trainable.

    Args:
        model: Base Whisper model

    Returns:
        PeftModel with LoRA adapters applied
    """
    # Create LoRA configuration
    # NOTE: Do NOT use task_type=TaskType.SEQ_2_SEQ_LM here.
    # SEQ_2_SEQ_LM wraps the model in PeftModelForSeq2SeqLM, whose forward()
    # injects an 'input_ids' kwarg for encoder text input (designed for T5/BART).
    # Whisper uses 'input_features' (mel spectrograms), not 'input_ids', so
    # setting task_type=None uses the generic PeftModel wrapper that passes all
    # kwargs through to the underlying WhisperForConditionalGeneration unchanged.
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=None,
        modules_to_save=LORA_CONFIG.get("modules_to_save", None),
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)

    # Print parameter statistics
    model.print_trainable_parameters()

    # Detailed breakdown
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n📊 LoRA Parameter Summary:")
    print(f"   Total parameters:     {total_params:>12,}")
    print(f"   Trainable parameters: {trainable_params:>12,}")
    print(f"   Frozen parameters:    {frozen_params:>12,}")
    print(f"   Trainable %:          {100 * trainable_params / total_params:>11.4f}%")
    print(f"\n📋 LoRA Configuration:")
    print(f"   Rank (r):             {LORA_CONFIG['r']}")
    print(f"   Alpha:                {LORA_CONFIG['lora_alpha']}")
    print(f"   Scaling (alpha/r):    {LORA_CONFIG['lora_alpha'] / LORA_CONFIG['r']:.1f}x")
    print(f"   Dropout:              {LORA_CONFIG['lora_dropout']}")
    print(f"   Target modules:       {LORA_CONFIG['target_modules']}")
    if LORA_CONFIG.get("modules_to_save"):
        print(f"   Fully trained:        {LORA_CONFIG['modules_to_save']}")

    return model


def load_processor() -> WhisperProcessor:
    """
    Load Whisper processor with Indonesian language configuration.

    Returns:
        WhisperProcessor configured for Indonesian
    """
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE_FULL,
        task=TASK,
    )

    # Ensure tokenizer is configured correctly
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE_FULL, task=TASK)

    return processor


def create_compute_metrics(processor: WhisperProcessor):
    """
    Create metrics computation function for WER and CER.

    Args:
        processor: WhisperProcessor for decoding

    Returns:
        Function to compute metrics
    """
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Compute metrics
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    return compute_metrics


def create_training_arguments(
    output_dir: str,
    run_name: str,
    **kwargs,
) -> Seq2SeqTrainingArguments:
    """
    Create training arguments for Seq2SeqTrainer.

    Args:
        output_dir: Directory to save checkpoints
        run_name: Name for this training run
        **kwargs: Override default training arguments

    Returns:
        Seq2SeqTrainingArguments
    """
    args = TRAINING_ARGS.copy()
    args.update(kwargs)
    args["output_dir"] = output_dir
    args["run_name"] = run_name

    return Seq2SeqTrainingArguments(**args)


def create_trainer(
    model,
    processor: WhisperProcessor,
    training_args: Seq2SeqTrainingArguments,
    train_dataset,
    eval_dataset,
    data_collator,
    metrics_logger: MetricsLogger = None,
    experiment_name: Optional[str] = None,
) -> Seq2SeqTrainer:
    """
    Create Seq2SeqTrainer for Whisper LoRA fine-tuning.

    Args:
        model: Whisper model with LoRA adapters (PeftModel)
        processor: WhisperProcessor
        training_args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        metrics_logger: MetricsLogger instance for comprehensive logging

    Returns:
        Seq2SeqTrainer instance
    """
    compute_metrics = create_compute_metrics(processor)

    # Early stopping callback
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_CONFIG["patience"],
            early_stopping_threshold=EARLY_STOPPING_CONFIG["threshold"],
        )
    ]

    # Add metrics logging callbacks if logger is provided
    if metrics_logger is not None:
        metrics_callbacks = create_metrics_callbacks(
            metrics_logger=metrics_logger,
            processor=processor,
            eval_dataset=eval_dataset,
            log_predictions=METRICS_LOGGING_CONFIG.get("log_predictions", True),
            num_prediction_samples=METRICS_LOGGING_CONFIG.get("num_prediction_samples", 5),
        )
        callbacks.extend(metrics_callbacks)

        # Log model configuration
        model_config = {
            "model_name": MODEL_NAME,
            "language": LANGUAGE,
            "language_full": LANGUAGE_FULL,
            "task": TASK,
            "strategy": "LoRA (PEFT)",
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "lora_dropout": LORA_CONFIG["lora_dropout"],
            "lora_target_modules": LORA_CONFIG["target_modules"],
            "lora_modules_to_save": LORA_CONFIG.get("modules_to_save", []),
            "dropout": MODEL_DROPOUT_CONFIG["dropout"],
            "attention_dropout": MODEL_DROPOUT_CONFIG["attention_dropout"],
            "activation_dropout": MODEL_DROPOUT_CONFIG["activation_dropout"],
            "num_beams": GENERATION_CONFIG["num_beams"],
            "max_length": GENERATION_CONFIG["max_length"],
        }
        metrics_logger.log_model_config(model_config)

    # --- Weighted Cross-Entropy ---
    token_weights = None
    if WEIGHTED_CE_CONFIG.get("enabled", False):
        pred_dirs = WEIGHTED_CE_CONFIG.get("predictions_dirs", [])
        existing_dirs = [d for d in pred_dirs if Path(d).exists()]
        if existing_dirs:
            print("\n⚖️  Building weighted CE token weights from ALL previous predictions...")
            token_weights = build_token_weights(
                predictions_dirs=existing_dirs,
                tokenizer=processor.tokenizer,
                num_folds=5,
                base_weight=WEIGHTED_CE_CONFIG.get("base_weight", 1.0),
                max_weight=WEIGHTED_CE_CONFIG.get("max_weight", 3.0),
                min_error_count=WEIGHTED_CE_CONFIG.get("min_error_count", 3),
                smoothing=WEIGHTED_CE_CONFIG.get("smoothing", 0.5),
                model_vocab_size=model.config.vocab_size,
                exclude_run_id=experiment_name,  # 🔒 Temporal Separation (Anti-Leakage)
            )
            print(f"   ✅ Token weights ready — {(token_weights > WEIGHTED_CE_CONFIG.get('base_weight', 1.0)).sum().item()} boosted tokens")
        else:
            print(f"\n⚠️  Weighted CE enabled but no predictions_dirs found: {pred_dirs}")
            print("   → Falling back to standard CE loss")

    trainer = WhisperLoRATrainer(
        token_weights=token_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=callbacks,
    )

    return trainer


def save_all_predictions(
    trainer: Seq2SeqTrainer,
    eval_dataset,
    processor: WhisperProcessor,
    save_path: str,
) -> None:
    """
    Run inference on the FULL eval dataset and save all predictions.
    This enables post-processing on any run without needing model checkpoints.

    Saves to: <metrics_dir>/all_predictions.json
    Format: [{"prediction": str, "reference": str}, ...]
    """
    import json
    from pathlib import Path

    print(f"  💾 Saving all {len(eval_dataset)} eval predictions...")

    predict_output = trainer.predict(eval_dataset)
    pred_ids = predict_output.predictions
    label_ids = predict_output.label_ids

    # Replace -100 with pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    all_predictions = [
        {"prediction": pred, "reference": ref}
        for pred, ref in zip(pred_strs, label_strs)
    ]

    save_file = Path(save_path) / "all_predictions.json"
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print(f"     Saved {len(all_predictions)} predictions → {save_file}")


def train_fold(
    fold_idx: int,
    train_dataset,
    eval_dataset,
    processor: WhisperProcessor,
    data_collator,
    output_dir: str,
    experiment_name: str = None,
) -> Dict[str, float]:
    """
    Train a single fold with LoRA and comprehensive metrics logging.

    Args:
        fold_idx: Fold index (0-4)
        train_dataset: Training dataset for this fold
        eval_dataset: Evaluation dataset for this fold
        processor: WhisperProcessor
        data_collator: Data collator
        output_dir: Base output directory
        experiment_name: Name for the experiment

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING FOLD {fold_idx + 1}/5 (LoRA)")
    print(f"{'=' * 60}")

    # CRITICAL: Re-initialize model for each fold to prevent weight leakage
    print("Loading fresh model (re-initialized from pretrained weights)...")
    model = load_model()

    # Apply LoRA adapters instead of freezing encoder
    print("\nApplying LoRA adapters...")
    model = apply_lora(model)

    # Log language configuration
    print(f"\n🌐 Language Configuration:")
    print(f"   Language Token: {LANGUAGE} (Indonesian as proxy for Minangkabau)")
    print(f"   Task: {TASK}")
    print(f"   Beam Search: {GENERATION_CONFIG['num_beams']} beams")

    # Create metrics logger for this fold
    metrics_logger = create_metrics_logger(
        output_dir=str(OUTPUT_DIR),
        fold_idx=fold_idx,
        experiment_name=experiment_name,
        save_locally=METRICS_LOGGING_CONFIG.get("save_locally", True),
        log_to_wandb=METRICS_LOGGING_CONFIG.get("log_to_wandb", True),
    )

    # Log dataset info
    metrics_logger.log_dataset_info(
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
    )

    # Create training arguments
    fold_output_dir = f"{output_dir}/fold_{fold_idx}"
    training_args = create_training_arguments(
        output_dir=fold_output_dir,
        run_name=f"lora-fold-{fold_idx}",
    )

    # Create trainer with metrics logger
    trainer = create_trainer(
        model=model,
        processor=processor,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        metrics_logger=metrics_logger,
        experiment_name=experiment_name,
    )

    # Train
    print(f"\n🚀 Starting LoRA training for fold {fold_idx + 1}...")
    trainer.train()

    # Evaluate
    print(f"\n📊 Evaluating fold {fold_idx + 1}...")
    eval_results = trainer.evaluate()

    print(f"\n✅ Fold {fold_idx + 1} Results:")
    print(f"   WER: {eval_results['eval_wer']:.4f} ({eval_results['eval_wer'] * 100:.2f}%)")
    print(f"   CER: {eval_results['eval_cer']:.4f} ({eval_results['eval_cer'] * 100:.2f}%)")

    # Save ALL eval predictions for post-processing
    save_all_predictions(
        trainer=trainer,
        eval_dataset=eval_dataset,
        processor=processor,
        save_path=str(metrics_logger.metrics_dir),
    )

    # Create and save summary report
    summary = metrics_logger.create_summary_report()
    print(f"\n📊 Metrics saved to: {metrics_logger.metrics_dir}")

    return {
        "wer": eval_results["eval_wer"],
        "cer": eval_results["eval_cer"],
        "metrics_dir": str(metrics_logger.metrics_dir),
    }


if __name__ == "__main__":
    print("Testing LoRA Trainer Module")
    print("-" * 50)

    # Load model and processor
    processor = load_processor()
    model = load_model()

    print(f"\nModel: {MODEL_NAME}")
    print(f"Language: {LANGUAGE_FULL} ({LANGUAGE})")
    print(f"Task: {TASK}")

    # Test LoRA application
    print("\nApplying LoRA adapters...")
    model = apply_lora(model)
