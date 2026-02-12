"""
Configuration for Post-Processing module.
Applies lexical correction to ASR predictions using a Minangkabau dictionary
built from training set transcripts.

Post-processing is applied AFTER model inference — results are saved separately
from raw model outputs for fair comparison.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PP_DIR = Path(__file__).parent               # code/post_processing/
CODE_DIR = PP_DIR.parent                     # code/
DATA_DIR = CODE_DIR / "Data"

# Output directories for post-processed results
OUTPUT_DIR = PP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# SOURCE RUNS — Base directories where training runs are stored
# =============================================================================
FE_METRICS_BASE = CODE_DIR / "freeze_encoder" / "outputs" / "metrics"
LORA_METRICS_BASE = CODE_DIR / "lora" / "outputs" / "metrics"

# Metadata for building dictionary
METADATA_CSV = DATA_DIR / "metadata_minang_wav.csv"
CSV_COLUMNS = ["audio_path", "language_code", "speaker_id", "transcript", "wav_path"]

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
NUM_FOLDS = 5
RANDOM_STATE = 42  # Must match training config for same fold splits

# =============================================================================
# DICTIONARY CONFIGURATION
# =============================================================================
DICTIONARY_CONFIG = {
    "max_edit_distance": 1,       # Maximum Levenshtein distance for correction
    "min_word_length": 5,         # Only correct words with ≥5 characters
    "min_word_frequency": 1,      # Include all words from training set
}

# =============================================================================
# TEXT NORMALIZATION
# =============================================================================
NORMALIZATION_CONFIG = {
    "lowercase": True,
    "remove_punctuation": True,
    "normalize_whitespace": True,
    "strip_leading_space": True,  # Whisper often adds leading space
}


# =============================================================================
# RUN DISCOVERY — Find available runs with all_predictions.json
# =============================================================================

def discover_runs(method: str) -> list:
    """
    Discover available training runs that have all_predictions.json saved.

    Args:
        method: 'freeze_encoder' or 'lora'

    Returns:
        List of (run_name, run_path) tuples, sorted newest first
    """
    if method == "freeze_encoder":
        base = FE_METRICS_BASE
    elif method == "lora":
        base = LORA_METRICS_BASE
    else:
        raise ValueError(f"Unknown method: {method}")

    if not base.exists():
        return []

    runs = []
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        # Check if at least fold_0/all_predictions.json exists
        pred_file = run_dir / "fold_0" / "all_predictions.json"
        if pred_file.exists():
            runs.append((run_dir.name, run_dir))

    return runs


def get_latest_run(method: str) -> tuple:
    """
    Get the latest run with all_predictions.json for a method.

    Returns:
        (run_name, run_path) or (None, None) if no runs found
    """
    runs = discover_runs(method)
    if runs:
        return runs[0]
    return None, None
