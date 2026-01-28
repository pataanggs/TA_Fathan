"""
Data loading module for Whisper fine-tuning.
Handles CSV loading, filtering for Minangkabau, and dataset merging.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
from datasets import Dataset

from config import (
    TRAIN_METADATA,
    TEST_METADATA,
    AUDIO_ROOT,
    CSV_COLUMNS,
    DATA_LANGUAGE,
)

# Pre-converted metadata file with WAV paths
MINANG_WAV_METADATA = AUDIO_ROOT / "metadata_minang_wav.csv"


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV metadata file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with audio metadata
    """
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    return df


def filter_by_language(df: pd.DataFrame, language_code: str = DATA_LANGUAGE) -> pd.DataFrame:
    """
    Filter dataframe to only include rows with specified language code.
    
    Args:
        df: Input dataframe
        language_code: Language code to filter by (default: 'min' for Minangkabau)
        
    Returns:
        Filtered dataframe
    """
    filtered_df = df[df["language_code"] == language_code].copy()
    print(f"Filtered to {len(filtered_df)} samples for language: {language_code}")
    return filtered_df


def create_full_path(df: pd.DataFrame, audio_root: Path = AUDIO_ROOT) -> pd.DataFrame:
    """
    Create full audio file paths by joining audio root with relative paths.
    
    Args:
        df: DataFrame with audio_path column
        audio_root: Root directory for audio files
        
    Returns:
        DataFrame with updated full paths
    """
    df = df.copy()
    df["full_path"] = df["audio_path"].apply(lambda x: str(audio_root / x))
    return df


def load_minang_wav_metadata() -> pd.DataFrame:
    """
    Load pre-converted Minangkabau WAV metadata.
    This is the preferred method if audio has been pre-converted.
    
    Returns:
        DataFrame with Minangkabau samples and WAV paths
    """
    if not MINANG_WAV_METADATA.exists():
        return None
    
    # CSV format: audio_path, language_code, speaker_id, transcript, wav_path
    columns = ["audio_path", "language_code", "speaker_id", "transcript", "wav_path"]
    df = pd.read_csv(MINANG_WAV_METADATA, header=None, names=columns)
    
    # Create full path using the wav_path column
    df["full_path"] = df["wav_path"].apply(lambda x: str(AUDIO_ROOT / x))
    
    print(f"✅ Loaded {len(df)} samples from pre-converted metadata")
    return df


def load_and_merge_datasets() -> pd.DataFrame:
    """
    Load train and test CSV files, filter for Minangkabau,
    and merge into a single DataFrame.
    
    Returns:
        Merged DataFrame with all Minangkabau samples
    """
    # Check for pre-converted metadata first
    if MINANG_WAV_METADATA.exists():
        print(f"Using pre-converted metadata: {MINANG_WAV_METADATA}")
        return load_minang_wav_metadata()
    
    # Fall back to original metadata
    print(f"Loading train metadata from: {TRAIN_METADATA}")
    train_df = load_metadata(TRAIN_METADATA)
    print(f"  Total train samples: {len(train_df)}")
    
    print(f"Loading test metadata from: {TEST_METADATA}")
    test_df = load_metadata(TEST_METADATA)
    print(f"  Total test samples: {len(test_df)}")
    
    # Filter for Minangkabau language
    train_df = filter_by_language(train_df)
    test_df = filter_by_language(test_df)
    
    # Create full paths
    train_df = create_full_path(train_df)
    test_df = create_full_path(test_df)
    
    # Add split column for reference
    train_df["original_split"] = "train"
    test_df["original_split"] = "test"
    
    # Concatenate into single DataFrame
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"\nTotal Minangkabau samples after merging: {len(merged_df)}")
    
    return merged_df


def dataframe_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert pandas DataFrame to HuggingFace Dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        HuggingFace Dataset object
    """
    # Select relevant columns
    dataset_df = df[["full_path", "transcript"]].copy()
    
    # Add speaker_id if available
    if "speaker_id" in df.columns:
        dataset_df["speaker_id"] = df["speaker_id"]
    
    dataset_df = dataset_df.rename(columns={
        "full_path": "audio_path",
        "transcript": "text",
    })
    
    dataset = Dataset.from_pandas(dataset_df)
    return dataset


def get_train_eval_split(
    dataset: Dataset, 
    train_indices: list, 
    eval_indices: list
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and eval based on indices.
    
    Args:
        dataset: Full HuggingFace Dataset
        train_indices: Indices for training
        eval_indices: Indices for evaluation
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices)
    return train_dataset, eval_dataset


if __name__ == "__main__":
    # Test data loading
    merged_df = load_and_merge_datasets()
    print("\nSample data:")
    print(merged_df.head())
    print(f"\nColumn dtypes:\n{merged_df.dtypes}")
