"""Configuration for Whisper LoRA Fine-tuning on Minangkabau language.
OPTIMIZED FOR TINY DATASET (156 Files / 1.3 Hours).
Strategy: Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).

Version history:
    v1 (17.32% WER): r=8,  alpha=16, LR=5e-4,   WD=0.01, LS=0.1, batch=32, steps=400
    v2 (18.03% WER): r=16, alpha=32, LR=3e-4,   WD=0.03, LS=0.05, batch=32, steps=250
    v3 (17.87% WER): r=8,  alpha=16, LR=7e-4,   WD=0.05, LS=0.1,  batch=32, steps=200
    v4 (18.20% WER): r=8,  alpha=16, LR=1.4e-3, WD=0.05, LS=0.1,  batch=64, steps=100

v5 Fixes (addressing v4 failure modes):
    Problem 1 — Batch/dataset mismatch:
        Effective batch 64 on ~125 files/fold = only 2 steps per epoch.
        AdamW needs more frequent updates to build reliable momentum.
        Fix: batch_size=16, grad_accum=1 (eff. batch=16), max_steps=400.

    Problem 2 — LR × Weighted-CE collision:
        max_weight=3.0 amplifies gradients 3× on hard tokens.
        LR=1.4e-3 × weight=3.0 risks gradient explosion on LoRA weights.
        Fix: lower LR back to 5e-4 (stable zone for r=16 with WCE).

    Problem 3 — LoRA capacity dilution:
        r=8 spread across 6 module types = too little capacity per layer.
        Fix: increase to r=16, alpha=32 to support all target modules.

    Problem 4 — SpecAugment blinding:
        time_mask=80 blocks large audio chunks; model can't hear clean phonemes
        in only ~400 short steps.
        Fix: reduce specaugment_time_mask to 50.

Applies LoRA to all attention + feed-forward linear layers.

References:
    - Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
      ICLR. DOI: 10.48550/arXiv.2106.09685
    - Yadav et al. (2025). Optimizer-Aware Fine-Tuning of Whisper Small with
      Low-Rank Adaption. Information, 16(11), 928. DOI: 10.3390/info16110928
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
WANDB_GROUP = "whisper-minang-lora-v5"

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
    "r": 16,                         # Increased from 8: r=8 across 6 modules was under-capacity
    "lora_alpha": 32,                # Scaling factor (alpha/r = 2x scaling, consistent with v2)
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
    "per_device_train_batch_size": 16,   # Reduced: eff. batch 64 was too large for ~125 samples/fold
    "per_device_eval_batch_size": 64,    # Max out eval throughput
    "gradient_accumulation_steps": 1,    # No accumulation: eff. batch = 16, more frequent updates
    "learning_rate": 5e-4,               # Lowered from 1.4e-3: safer with max_weight=3.0 in Weighted CE
    "warmup_ratio": 0.1,                 # 10% warmup (consistent with v1/v3)
    "max_steps": 400,                    # 4× increase to match total data exposure at eff. batch=16
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
    "label_smoothing_factor": 0.1,       # Critical for fold consistency on tiny data
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
    "specaugment_time_mask": 50,             # Reduced from 80: heavy masking blinds model on tiny data
    "specaugment_freq_mask": 40,             # Kept at 40 — moderate frequency masking
    "pitch_shift": 2,                        # Restored from 3 — moderate pitch range
}

CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript"]
