"""
Configuration for Freeze Encoder (Decoder-Only) Fine-tuning strategy.
Whisper Base on Minangkabau language.
OPTIMIZED FOR TINY DATASET (156 Files / 1.3 Hours).

Strategy: Freeze entire encoder, train full decoder (~37M params, 50%).

Version history:
    v1 (19.57% WER): batch=64, LR=2e-5, WD=0.1,  dropout=0.2, time_mask=80, steps=60
    v2          ---: batch=16, LR=1e-5, WD=0.05, dropout=0.1, time_mask=20, steps=400

v3 Fixes (addressing v2 failure modes):
    Problem 1 — Conservative LR with unstable single-sample updates:
        LR=1e-5 with batch_size=1 per step (eff. batch=16 via single accum) means
        each individual step is high-variance. Using grad_accum=2 smooths gradient
        estimates before each update, allowing a slightly higher LR to be safe.
        Fix: batch=8, grad_accum=2 (eff. batch=16 preserved), LR=2e-5.

    Problem 2 — Weight decay still constraining tiny-data learning:
        WD=0.05 + dropout=0.1 + label_smoothing=0.1 is still a triple penalty.
        With only 1.3h of data the decoder needs maximum freedom to absorb
        Minangkabau patterns from very few examples.
        Fix: weight_decay=0.01 (minimal L2, near-zero constraint).

    Problem 3 — SpecAugment masking still corrupting frozen-encoder output:
        Even time_mask=20 can cut 20 mel-frames (~200ms) from a short clip;
        the decoder still receives some corrupted input. At this stage, get
        the baseline solid before reintroducing masking.
        Fix: time_mask=10, freq_mask=10 (minimal regularization floor).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# PATHS - resolve relative to code/ root
# =============================================================================
STRATEGY_DIR = Path(__file__).parent          # code/freeze_encoder/
CODE_DIR = STRATEGY_DIR.parent                # code/

load_dotenv(CODE_DIR / ".env")

BASE_DIR = CODE_DIR
DATA_DIR = BASE_DIR / "Data"
TRAIN_METADATA = DATA_DIR / "metadata_train.csv"
TEST_METADATA = DATA_DIR / "metadata_test.csv"
AUDIO_ROOT = DATA_DIR

OUTPUT_DIR = STRATEGY_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PROCESSED_AUDIO_DIR = OUTPUT_DIR / "processed_audio"

OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
PROCESSED_AUDIO_DIR.mkdir(exist_ok=True)

# =============================================================================
# WANDB
# =============================================================================
WANDB_API_KEY = os.getenv("API_KEY")
WANDB_PROJECT = "whisper-minangkabau"
WANDB_GROUP = "whisper-minang-freeze-encoder-v3"

# =============================================================================
# MODEL
# =============================================================================
MODEL_NAME = "openai/whisper-base"
LANGUAGE = "id"
LANGUAGE_FULL = "indonesian"
DATA_LANGUAGE = "min"
TASK = "transcribe"
FREEZE_ENCODER = True

# =============================================================================
# AUDIO CONFIGURATION
# =============================================================================
SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0

# =============================================================================
# TRAINING ARGS
# =============================================================================
NUM_FOLDS = 5
RANDOM_STATE = 42

TRAINING_ARGS = {
    "output_dir": str(CHECKPOINT_DIR),
    "per_device_train_batch_size": 16,       
    "per_device_eval_batch_size": 64,        
    "gradient_accumulation_steps": 1,        # Effective batch=16 preserved; smoother gradient estimate per update
    "learning_rate": 2e-5,                   # Slightly raised: grad_accum smoothing makes this safe
    "warmup_ratio": 0.1,                     # 10% warmup (lower now that steps=400)
    "max_steps": 400,                        # 4x increase: AdamW needs time to build momentum
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "gradient_checkpointing": False,         # Disabled: trade VRAM for ~30% speed
    "bf16": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "weight_decay": 0.01,                    # Near-zero: triple-penalty (WD+dropout+LS) still too tight on 1.3h data
    "max_grad_norm": 1.0,                    # Restored to default (0.5 was too aggressive)
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
    "label_smoothing_factor": 0.1,           # Prevent overconfident predictions
}

# DECODER DROPOUT — standard Whisper defaults (over-regularization removed)
MODEL_DROPOUT_CONFIG = {
    "dropout": 0.1,               # Whisper default: triple-penalty (WD+dropout+LS) caused underfitting
    "attention_dropout": 0.1,     # Whisper default
    "activation_dropout": 0.1,    # Whisper default
}

EARLY_STOPPING_CONFIG = {
    "patience": 6,        # Tighter: stop sooner to prevent overtraining
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
# AUGMENTATION (CRITICAL FOR TINY DATA)
# =============================================================================
AUGMENTATION_CONFIG = {
    "speed_perturbation": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
    "noise_snr_range": (10, 30),
    "specaugment_time_mask": 10,             # Minimal floor: frozen encoder still can't recover masked signal;
                                             #   establish clean baseline before reintroducing masking
    "specaugment_freq_mask": 10,             # Minimal floor — same rationale
    "pitch_shift": 2,                        # Restored from 3 — moderate pitch range
}

CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript"]
