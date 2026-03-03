"""Configuration for Whisper LoRA Fine-tuning on Minangkabau language.
OPTIMIZED FOR TINY DATASET (156 Files / 1.3 Hours).
Strategy: Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

v3 Sweet-Spot (based on v1 & v2 comparison):
    v1 (17.32% WER): r=8, LR=5e-4, WD=0.01, LS=0.1, dropout=0.1/0.05/0.05
    v2 (18.03% WER): r=16, LR=3e-4, WD=0.03, LS=0.05, dropout=0.15/0.1/0.1
    → r=16 too many params for 1.3h; LR=3e-4 too weak for few LoRA params
    → LS=0.05 hurt fold stability (std 1.01% vs 0.83%)

    v3 strategy:
    - Revert r=8, alpha=16 (low-rank = structural regularization for tiny data)
    - Raise LR to 7e-4 (LoRA params need stronger gradient signal)
    - Restore label_smoothing=0.1 (critical for fold consistency)
    - Increase weight_decay to 0.05 (midpoint: stronger than v1's 0.01)
    - Keep lora_dropout=0.15 (v2 showed this helps vs v1's 0.1)
    - Revert warmup_ratio=0.1 (v1 was fine; 0.15 wastes steps)
    - Revert model dropout to v1 levels (0.1/0.05/0.05)

Applies LoRA to all attention + feed-forward linear layers.

References:
    - Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
    - Yadav et al. (2025). Optimizer-Aware Fine-Tuning of Whisper Small with
      Low-Rank Adaption. Information, 16(11), 928.
    - Sharma et al. (2025). Fine-tuning Whisper Tiny for Swahili ASR. AfricanNLP.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATHS - resolve relative to code/ root
# =============================================================================
STRATEGY_DIR = Path(__file__).parent          # code/lora/
CODE_DIR = STRATEGY_DIR.parent                # code/

load_dotenv(CODE_DIR / ".env")

BASE_DIR = CODE_DIR
DATA_DIR = BASE_DIR / "Data"
TRAIN_METADATA = DATA_DIR / "metadata_train.csv"
TEST_METADATA = DATA_DIR / "metadata_test.csv"
AUDIO_ROOT = DATA_DIR

OUTPUT_DIR = STRATEGY_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# =============================================================================
# WANDB
# =============================================================================
WANDB_API_KEY = os.getenv("API_KEY")
WANDB_PROJECT = "whisper-minangkabau"
WANDB_GROUP = "whisper-minang-lora-v1"

# =============================================================================
# MODEL
# =============================================================================
MODEL_NAME = "openai/whisper-base"
LANGUAGE = "id"
LANGUAGE_FULL = "indonesian"
DATA_LANGUAGE = "min"
TASK = "transcribe"
FREEZE_ENCODER = False  # LoRA doesn't use freeze encoder strategy

# =============================================================================
# AUDIO CONFIGURATION (same as Freeze Encoder for fair comparison)
# =============================================================================
SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0

# =============================================================================
# LoRA CONFIGURATION (PEFT)
# =============================================================================
LORA_CONFIG = {
    "r": 8,                          # Low rank = structural regularization for 1.3h data
    "lora_alpha": 16,                # Scaling factor (alpha/r = 2x scaling)
    "lora_dropout": 0.15,            # Dropout on LoRA layers (↑ from v1's 0.1)
    "target_modules": [              # All attention + FFN linear layers
        "q_proj",                    # Query projection
        "v_proj",                    # Value projection
        "k_proj",                    # Key projection
        "out_proj",                  # Output projection
        "fc1",                       # Feed-forward layer 1
        "fc2",                       # Feed-forward layer 2
    ],
    "bias": "none",                  # Don't train bias parameters
    "task_type": "SEQ_2_SEQ_LM",    # Sequence-to-sequence language modeling
    "modules_to_save": [],           
}

# =============================================================================
# TRAINING ARGS
# =============================================================================
NUM_FOLDS = 5
RANDOM_STATE = 42

TRAINING_ARGS = {
    "output_dir": str(CHECKPOINT_DIR),
    "per_device_train_batch_size": 32,   
    "per_device_eval_batch_size": 64,    # Max out eval throughput
    "gradient_accumulation_steps": 2,    # Effective batch size = 64 (doubled from 32)
    "learning_rate": 1.4e-3,             # 2x from 7e-4 (linear scaling with batch)
    "warmup_ratio": 0.1,                 # 10% warmup (v1 baseline was stable)
    "max_steps": 100,                    # Halved (effective batch doubled = same data seen)
    "lr_scheduler_type": "cosine",       # Cosine annealing
    "optim": "adamw_torch",
    "gradient_checkpointing": False,     
    "bf16": True,                        
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "weight_decay": 0.05,               # Midpoint between v1 (0.01) and FE (0.1)
    "eval_strategy": "steps",
    "eval_steps": 4,
    "save_steps": 4,
    "logging_steps": 1,
    "logging_first_step": True,
    "load_best_model_at_end": True,
    "metric_for_best_model": "wer",
    "greater_is_better": False,
    "save_total_limit": 1,
    "report_to": "wandb",
    "push_to_hub": False,
    "predict_with_generate": True,
    "generation_max_length": 225,
    "torch_compile": False,
    "label_smoothing_factor": 0.1,       # Restored — critical for fold consistency on tiny data
}

# LoRA has implicit regularization, so lower dropout than Freeze Encoder
MODEL_DROPOUT_CONFIG = {
    "dropout": 0.1,               # v1 level (LoRA already constrains capacity)
    "attention_dropout": 0.05,    # Light attention regularization
    "activation_dropout": 0.05,   # Light activation regularization
}

EARLY_STOPPING_CONFIG = {
    "patience": 6,      # Tighter patience to prevent overtraining past optimum
    "threshold": 0.001,
}

# =============================================================================
# METRICS LOGGING CONFIGURATION
# =============================================================================
METRICS_DIR = OUTPUT_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

METRICS_LOGGING_CONFIG = {
    "save_locally": True,
    "log_to_wandb": True,
    "log_predictions": True,
    "num_prediction_samples": 5,
    "log_predictions_every_n_evals": 1,
}

GENERATION_CONFIG = {
    "num_beams": 5,              # Beam search: explores 5 paths for better WER
    "max_length": 225,
    "language": LANGUAGE,
    "task": TASK,
}

# =============================================================================
# WEIGHTED CROSS-ENTROPY CONFIGURATION
# Analyzes previous run predictions to boost loss on error-prone tokens.
# Set "enabled" to False to use standard CE loss.
# =============================================================================
WEIGHTED_CE_CONFIG = {
    "enabled": True,
    # List of metrics directories to scan for predictions.
    # Auto-discovers ALL run subdirectories — aggregates error statistics across every run.
    "predictions_dirs": [
        str(CODE_DIR / "lora" / "outputs" / "metrics"),
        str(CODE_DIR / "freeze_encoder" / "outputs" / "metrics"),
    ],
    "base_weight": 1.0,          # Default weight for normal tokens
    "max_weight": 3.0,           # Maximum weight for most error-prone tokens
    "min_error_count": 3,        # Minimum errors to apply boosted weight (noise filter)
    "smoothing": 0.5,            # Sigmoid steepness (lower = smoother transition)
}

# =============================================================================
# AUGMENTATION (same as Freeze Encoder for fair comparison)
# =============================================================================
AUGMENTATION_CONFIG = {
    "speed_perturbation": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
    "noise_snr_range": (10, 30),
    "specaugment_time_mask": 80,             # Restored from 100 — less aggressive masking
    "specaugment_freq_mask": 40,             # Restored from 50
    "pitch_shift": 2,                        # Restored from 3 — moderate pitch range
}

CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript"]
