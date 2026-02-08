"""
Configuration for Freeze Encoder (Decoder-Only) Fine-tuning strategy.
Whisper Base on Minangkabau language.
OPTIMIZED FOR TINY DATASET (156 Files / 1.3 Hours).
Focus: Anti-Overfitting & Aggressive Augmentation.

Strategy: Freeze entire encoder, train full decoder (~37M params, 50%).
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
WANDB_GROUP = "whisper-minang-freeze-encoder-v1"

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
    "per_device_train_batch_size": 32,       # Increased (16GB VRAM headroom)
    "per_device_eval_batch_size": 64,        # Max out eval throughput
    "gradient_accumulation_steps": 1,        # Effective batch = 32
    "learning_rate": 1e-5,
    "warmup_ratio": 0.2,                     # Higher warmup (20%)
    "max_steps": 200,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_torch",
    "gradient_checkpointing": False,         # Disabled: trade VRAM for ~30% speed
    "bf16": True,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "weight_decay": 0.1,                     # Reduced from 0.2 (risk of underfitting
                                              #   combined with 0.3 dropout)
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
}

# ANTI-OVERFITTING DROPOUT
MODEL_DROPOUT_CONFIG = {
    "dropout": 0.3,
    "attention_dropout": 0.2,
    "activation_dropout": 0.2,
}

EARLY_STOPPING_CONFIG = {
    "patience": 8,        # Stop if no improvement after ~1 epoch (8 steps)
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
# AUGMENTATION (CRITICAL FOR TINY DATA)
# =============================================================================
AUGMENTATION_CONFIG = {
    "speed_perturbation": [0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15],
    "noise_snr_range": (10, 30),
    "specaugment_time_mask": 80,
    "specaugment_freq_mask": 40,
    "pitch_shift": 2,
}

CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript"]
