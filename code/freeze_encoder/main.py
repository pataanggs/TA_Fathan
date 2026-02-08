"""
Main entry point for Whisper fine-tuning on Minangkabau language.
Strategy: Freeze Encoder (Decoder-Only Fine-Tuning).
Implements 5-fold cross-validation with WandB integration.
Includes comprehensive metrics logging to both local files and WandB.

Usage:
    cd /path/to/code/freeze_encoder
    python main.py
"""

import os
import sys
import warnings
import logging
import json
import numpy as np
import torch
import wandb
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import KFold

# Suppress all warnings BEFORE importing transformers
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# =============================================================================
# PATH SETUP: Ensure this strategy's config is used by shared modules
# =============================================================================
STRATEGY_DIR = Path(__file__).parent          # code/freeze_encoder/
CODE_DIR = STRATEGY_DIR.parent                # code/

# 1. Import our own config first
sys.path.insert(0, str(STRATEGY_DIR))
import config  # This imports freeze_encoder/config.py

# 2. Register it as THE config module so shared modules use it
sys.modules['config'] = config

# 3. Add parent (code/) to path for shared modules
sys.path.insert(0, str(CODE_DIR))

from config import (
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_GROUP,
    NUM_FOLDS,
    RANDOM_STATE,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    TRAINING_ARGS,
    MODEL_DROPOUT_CONFIG,
    AUGMENTATION_CONFIG,
    METRICS_LOGGING_CONFIG,
)

# Import shared modules from parent (code/) directory
from data_loader import (
    load_and_merge_datasets,
    dataframe_to_hf_dataset,
    get_train_eval_split,
)
from dataset import (
    prepare_dataset_for_training,
    prepare_dataset_for_evaluation,
    create_data_collator,
)
from augmentation import AudioAugmenter

# Import strategy-specific trainer
from trainer import (
    load_processor,
    load_model,
    train_fold,
)


def check_gpu():
    """Check and display GPU information."""
    print("\n[GPU CHECK]")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ No GPU available - training will use CPU (very slow)")
        return False


def setup_wandb():
    """Initialize WandB with API key from .env"""
    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        print("✅ WandB API key loaded from .env")
    else:
        print("⚠️  Warning: WANDB_API_KEY not found in .env")


def print_experiment_config():
    """Print the full experiment configuration for reproducibility."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION (Freeze Encoder)")
    print("=" * 70)

    print("\n📋 Strategy: Freeze Encoder (Decoder-Only)")
    print(f"   Trainable: ~37M params (~50% of 74M)")

    print("\n📋 Training Configuration:")
    print(f"   Learning Rate:      {TRAINING_ARGS['learning_rate']}")
    print(f"   Batch Size:         {TRAINING_ARGS['per_device_train_batch_size']}")
    print(f"   Grad Accum Steps:   {TRAINING_ARGS['gradient_accumulation_steps']}")
    print(f"   Effective Batch:    {TRAINING_ARGS['per_device_train_batch_size'] * TRAINING_ARGS['gradient_accumulation_steps']}")
    print(f"   Max Steps:          {TRAINING_ARGS['max_steps']}")
    print(f"   Warmup Ratio:       {TRAINING_ARGS['warmup_ratio']}")
    print(f"   Weight Decay:       {TRAINING_ARGS['weight_decay']}")
    print(f"   LR Scheduler:       {TRAINING_ARGS['lr_scheduler_type']}")

    print("\n📋 Model Dropout (Anti-Overfitting):")
    print(f"   Dropout:            {MODEL_DROPOUT_CONFIG['dropout']}")
    print(f"   Attention Dropout:  {MODEL_DROPOUT_CONFIG['attention_dropout']}")
    print(f"   Activation Dropout: {MODEL_DROPOUT_CONFIG['activation_dropout']}")

    print("\n📋 Augmentation Config:")
    print(f"   Speed Perturbation: {AUGMENTATION_CONFIG['speed_perturbation']}")
    print(f"   Noise SNR Range:    {AUGMENTATION_CONFIG['noise_snr_range']}")
    print(f"   SpecAugment Time:   {AUGMENTATION_CONFIG['specaugment_time_mask']}")
    print(f"   SpecAugment Freq:   {AUGMENTATION_CONFIG['specaugment_freq_mask']}")
    print(f"   Pitch Shift:        {AUGMENTATION_CONFIG['pitch_shift']}")


def main():
    """Main training pipeline with 5-fold cross-validation."""
    
    print("=" * 70)
    print("WHISPER FINE-TUNING FOR MINANGKABAU LANGUAGE")
    print("Strategy: Freeze Encoder (Decoder-Only)")
    print("=" * 70)
    
    # Check GPU
    check_gpu()
    
    # Setup WandB
    setup_wandb()

    # Print full configuration
    print_experiment_config()
    
    # Create experiment name with timestamp
    experiment_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n📊 Experiment Name: {experiment_name}")
    print(f"   Metrics will be saved to: {OUTPUT_DIR / 'metrics' / experiment_name}")
    
    # ==========================================================================
    # STEP 1: Load data (uses pre-converted WAV metadata if available)
    # ==========================================================================
    print("\n[STEP 1] Loading data...")
    
    merged_df = load_and_merge_datasets()
    
    print(f"Final dataset size: {len(merged_df)} samples")
    
    # Convert to HuggingFace Dataset
    dataset = dataframe_to_hf_dataset(merged_df)
    
    # ==========================================================================
    # STEP 2: Setup processor and augmenter
    # ==========================================================================
    print("\n[STEP 2] Setting up processor and augmenter...")
    
    processor = load_processor()
    augmenter = AudioAugmenter()
    
    # Create a dummy model for data collator
    dummy_model = load_model()
    data_collator = create_data_collator(processor, dummy_model)
    del dummy_model  # Free memory
    torch.cuda.empty_cache()
    
    # ==========================================================================
    # STEP 3: 5-Fold Cross-Validation
    # ==========================================================================
    print("\n[STEP 3] Starting 5-Fold Cross-Validation...")
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    
    for fold_idx, (train_indices, eval_indices) in enumerate(kfold.split(range(len(dataset)))):
        
        # Initialize WandB for this fold
        wandb.init(
            project=WANDB_PROJECT,
            group=WANDB_GROUP,
            name=f"freeze-fold-{fold_idx}",
            reinit=True,
            config={
                "strategy": "Freeze Encoder",
                "fold": fold_idx,
                "learning_rate": TRAINING_ARGS["learning_rate"],
                "max_steps": TRAINING_ARGS["max_steps"],
                "weight_decay": TRAINING_ARGS["weight_decay"],
                "dropout": MODEL_DROPOUT_CONFIG["dropout"],
            },
        )
        
        print(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} (Freeze Encoder) ---")
        print(f"Train samples: {len(train_indices)}")
        print(f"Eval samples: {len(eval_indices)}")
        
        # Split dataset
        train_dataset, eval_dataset = get_train_eval_split(
            dataset, 
            train_indices.tolist(), 
            eval_indices.tolist()
        )
        
        # Prepare datasets
        print("Preparing training dataset with augmentation...")
        train_dataset_prepared = prepare_dataset_for_training(
            train_dataset, processor, augmenter
        )
        
        print("Preparing evaluation dataset...")
        eval_dataset_prepared = prepare_dataset_for_evaluation(
            eval_dataset, processor
        )
        
        # Train fold
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            train_dataset=train_dataset_prepared,
            eval_dataset=eval_dataset_prepared,
            processor=processor,
            data_collator=data_collator,
            output_dir=str(CHECKPOINT_DIR),
            experiment_name=experiment_name,
        )
        
        fold_results.append(fold_metrics)
        
        # Log to WandB
        wandb.log({
            "fold": fold_idx,
            "fold_wer": fold_metrics["wer"],
            "fold_cer": fold_metrics["cer"],
        })
        
        # Finish WandB run for this fold
        wandb.finish()

        # Clear GPU memory between folds
        torch.cuda.empty_cache()
    
    # ==========================================================================
    # STEP 4: Calculate and print final statistics
    # ==========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS - 5-FOLD CROSS-VALIDATION (Freeze Encoder)")
    print("=" * 70)
    
    wer_scores = [r["wer"] for r in fold_results]
    cer_scores = [r["cer"] for r in fold_results]
    
    print("\nPer-Fold Results:")
    print("-" * 50)
    for i, result in enumerate(fold_results):
        print(f"  Fold {i + 1}: WER = {result['wer']:.4f} ({result['wer']*100:.2f}%), "
              f"CER = {result['cer']:.4f} ({result['cer']*100:.2f}%)")
    
    print("\n" + "-" * 50)
    print(f"📊 Mean WER: {np.mean(wer_scores)*100:.2f}% ± {np.std(wer_scores)*100:.2f}%")
    print(f"📊 Mean CER: {np.mean(cer_scores)*100:.2f}% ± {np.std(cer_scores)*100:.2f}%")
    print(f"📊 Best WER: {np.min(wer_scores)*100:.2f}% (Fold {np.argmin(wer_scores) + 1})")
    print(f"📊 Best CER: {np.min(cer_scores)*100:.2f}% (Fold {np.argmin(cer_scores) + 1})")
    print("-" * 50)
    
    # ==========================================================================
    # STEP 5: Save comprehensive cross-validation summary locally
    # ==========================================================================
    cv_summary = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "strategy": "Freeze Encoder (Decoder-Only)",
        "num_folds": NUM_FOLDS,
        "training_config": {
            "learning_rate": TRAINING_ARGS["learning_rate"],
            "batch_size": TRAINING_ARGS["per_device_train_batch_size"],
            "gradient_accumulation_steps": TRAINING_ARGS["gradient_accumulation_steps"],
            "effective_batch_size": TRAINING_ARGS["per_device_train_batch_size"] * TRAINING_ARGS["gradient_accumulation_steps"],
            "max_steps": TRAINING_ARGS["max_steps"],
            "warmup_ratio": TRAINING_ARGS["warmup_ratio"],
            "weight_decay": TRAINING_ARGS["weight_decay"],
            "lr_scheduler_type": TRAINING_ARGS["lr_scheduler_type"],
        },
        "model_config": {
            "dropout": MODEL_DROPOUT_CONFIG["dropout"],
            "attention_dropout": MODEL_DROPOUT_CONFIG["attention_dropout"],
            "activation_dropout": MODEL_DROPOUT_CONFIG["activation_dropout"],
        },
        "augmentation_config": {
            "speed_perturbation": AUGMENTATION_CONFIG["speed_perturbation"],
            "noise_snr_range": list(AUGMENTATION_CONFIG["noise_snr_range"]),
            "specaugment_time_mask": AUGMENTATION_CONFIG["specaugment_time_mask"],
            "specaugment_freq_mask": AUGMENTATION_CONFIG["specaugment_freq_mask"],
            "pitch_shift": AUGMENTATION_CONFIG["pitch_shift"],
        },
        "per_fold_results": [
            {
                "fold": i,
                "wer": r["wer"],
                "cer": r["cer"],
                "wer_percent": r["wer"] * 100,
                "cer_percent": r["cer"] * 100,
                "metrics_dir": r.get("metrics_dir", ""),
            }
            for i, r in enumerate(fold_results)
        ],
        "cross_validation_summary": {
            "mean_wer": float(np.mean(wer_scores)),
            "std_wer": float(np.std(wer_scores)),
            "mean_cer": float(np.mean(cer_scores)),
            "std_cer": float(np.std(cer_scores)),
            "min_wer": float(np.min(wer_scores)),
            "max_wer": float(np.max(wer_scores)),
            "min_cer": float(np.min(cer_scores)),
            "max_cer": float(np.max(cer_scores)),
            "mean_wer_percent": float(np.mean(wer_scores) * 100),
            "std_wer_percent": float(np.std(wer_scores) * 100),
            "mean_cer_percent": float(np.mean(cer_scores) * 100),
            "std_cer_percent": float(np.std(cer_scores) * 100),
        }
    }
    
    # Save cross-validation summary locally
    cv_summary_dir = OUTPUT_DIR / "metrics" / experiment_name
    cv_summary_dir.mkdir(parents=True, exist_ok=True)
    cv_summary_path = cv_summary_dir / "cross_validation_summary.json"
    
    with open(cv_summary_path, 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"\n📊 Cross-validation summary saved to: {cv_summary_path}")
    
    # Log summary to WandB
    wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=f"{experiment_name}-summary",
        reinit=True,
    )
    wandb.log({
        "cv/mean_wer": np.mean(wer_scores),
        "cv/std_wer": np.std(wer_scores),
        "cv/mean_cer": np.mean(cer_scores),
        "cv/std_cer": np.std(cer_scores),
        "cv/min_wer": np.min(wer_scores),
        "cv/max_wer": np.max(wer_scores),
        "cv/min_cer": np.min(cer_scores),
        "cv/max_cer": np.max(cer_scores),
    })
    
    # Log per-fold results as a table
    fold_table = wandb.Table(
        columns=["Fold", "WER", "CER", "WER %", "CER %"],
        data=[
            [i+1, r["wer"], r["cer"], r["wer"]*100, r["cer"]*100]
            for i, r in enumerate(fold_results)
        ]
    )
    wandb.log({"cv/fold_results": fold_table})
    
    # Update summary
    wandb.summary.update({
        "strategy": "Freeze Encoder",
        "mean_wer": np.mean(wer_scores),
        "std_wer": np.std(wer_scores),
        "mean_cer": np.mean(cer_scores),
        "std_cer": np.std(cer_scores),
    })
    wandb.finish()
    
    print("\n✅ Freeze Encoder Training complete!")
    print(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"📊 All metrics saved to: {OUTPUT_DIR / 'metrics' / experiment_name}")
    
    return fold_results


if __name__ == "__main__":
    results = main()
