"""
Audio preprocessing module for Whisper fine-tuning.
Now simplified to load pre-converted WAV files at 16kHz.
"""

import librosa
import numpy as np

from config import SAMPLE_RATE


def load_wav_audio(audio_path: str) -> np.ndarray:
    """
    Load a WAV audio file and return as numpy array.
    Audio files should already be converted to 16kHz WAV format.
    
    Args:
        audio_path: Path to WAV file
        
    Returns:
        Audio array at 16kHz
    """
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return audio


def get_duration(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """
    Calculate audio duration in seconds.
    
    Args:
        audio: Audio array
        sample_rate: Sample rate
        
    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range.
    
    Args:
        audio: Input audio array
        
    Returns:
        Normalized audio array
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


# Alias for backward compatibility
load_processed_audio = load_wav_audio


if __name__ == "__main__":
    # Test loading WAV files
    from data_loader import load_and_merge_datasets
    
    df = load_and_merge_datasets()
    
    # Test with first few files
    sample_paths = df["full_path"].head(3).tolist()
    
    for path in sample_paths:
        print(f"\nLoading: {path}")
        audio = load_wav_audio(path)
        duration = get_duration(audio)
        print(f"  Duration: {duration:.2f}s")
        print(f"  Shape: {audio.shape}")
        print(f"  Sample rate: {SAMPLE_RATE} Hz")
