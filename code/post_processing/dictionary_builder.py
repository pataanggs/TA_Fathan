"""
Dictionary builder for post-processing.
Builds a word-frequency dictionary from training set transcripts.
Dictionary is built PER FOLD (only from training indices) to prevent data leakage.
"""

import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from sklearn.model_selection import KFold

from config import (
    METADATA_CSV,
    CSV_COLUMNS,
    NUM_FOLDS,
    RANDOM_STATE,
    DICTIONARY_CONFIG,
)


def load_all_transcripts() -> List[str]:
    """
    Load all transcripts from the metadata CSV.

    Returns:
        List of transcript strings
    """
    df = pd.read_csv(METADATA_CSV, header=None, names=CSV_COLUMNS)
    transcripts = df["transcript"].tolist()
    print(f"Loaded {len(transcripts)} transcripts from {METADATA_CSV.name}")
    return transcripts


def clean_text(text: str) -> str:
    """
    Clean text for dictionary building: lowercase, remove punctuation, normalize.

    Args:
        text: Raw transcript text

    Returns:
        Cleaned text
    """
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_words(text: str) -> List[str]:
    """
    Extract individual words from cleaned text.

    Args:
        text: Cleaned text

    Returns:
        List of words
    """
    min_len = DICTIONARY_CONFIG["min_word_length"]
    words = text.split()
    return [w for w in words if len(w) >= min_len]


def build_dictionary(transcripts: List[str]) -> Dict[str, int]:
    """
    Build word-frequency dictionary from a list of transcripts.

    Args:
        transcripts: List of transcript strings

    Returns:
        Dictionary mapping word -> frequency count
    """
    word_counter = Counter()

    for transcript in transcripts:
        cleaned = clean_text(transcript)
        words = extract_words(cleaned)
        word_counter.update(words)

    # Filter by minimum frequency
    min_freq = DICTIONARY_CONFIG["min_word_frequency"]
    dictionary = {
        word: count
        for word, count in word_counter.items()
        if count >= min_freq
    }

    return dictionary


def build_fold_dictionaries() -> Dict[int, Dict[str, int]]:
    """
    Build per-fold dictionaries using only training set transcripts.
    Uses the same KFold split as the training pipeline to prevent data leakage.

    Returns:
        Dictionary mapping fold_index -> word_frequency_dict
    """
    transcripts = load_all_transcripts()
    n_samples = len(transcripts)

    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_dictionaries = {}

    for fold_idx, (train_indices, eval_indices) in enumerate(kfold.split(range(n_samples))):
        # Build dictionary ONLY from training transcripts (no data leakage)
        train_transcripts = [transcripts[i] for i in train_indices]
        dictionary = build_dictionary(train_transcripts)

        fold_dictionaries[fold_idx] = dictionary

        print(
            f"  Fold {fold_idx}: {len(train_indices)} train samples "
            f"→ {len(dictionary)} unique words in dictionary"
        )

    return fold_dictionaries


def get_dictionary_words(dictionary: Dict[str, int]) -> Set[str]:
    """
    Get the set of words from a dictionary (for fast lookup).

    Args:
        dictionary: Word-frequency dictionary

    Returns:
        Set of dictionary words
    """
    return set(dictionary.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("DICTIONARY BUILDER — Minangkabau Post-Processing")
    print("=" * 60)

    print("\nBuilding per-fold dictionaries...")
    fold_dicts = build_fold_dictionaries()

    # Show sample words from fold 0
    print(f"\nSample words from Fold 0 dictionary:")
    sample_words = sorted(fold_dicts[0].items(), key=lambda x: -x[1])[:20]
    for word, freq in sample_words:
        print(f"  {word:20s} (freq: {freq})")

    # Show overlap statistics
    all_words = [set(d.keys()) for d in fold_dicts.values()]
    common = all_words[0].intersection(*all_words[1:])
    print(f"\nWords common to ALL folds: {len(common)}")
