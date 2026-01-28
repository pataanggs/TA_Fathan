"""
Script to pre-convert all audio files to WAV format.
Run this once before training.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import librosa
import soundfile as sf

from config import (
    TRAIN_METADATA,
    TEST_METADATA,
    AUDIO_ROOT,
    CSV_COLUMNS,
    LANGUAGE,
    SAMPLE_RATE,
)


def convert_audio_to_wav(input_path: str, output_path: str) -> bool:
    """Convert audio file to WAV format at 16kHz."""
    try:
        # Load audio
        audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
        
        # Normalize amplitude
        max_val = max(abs(audio.max()), abs(audio.min()))
        if max_val > 0:
            audio = audio / max_val
        
        # Save as WAV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, SAMPLE_RATE)
        return True
    except Exception as e:
        print(f"Error: {input_path} - {e}")
        return False


def main():
    print("=" * 60)
    print("AUDIO PRE-CONVERSION TO WAV")
    print("=" * 60)
    
    # Output directory for WAV files
    wav_dir = AUDIO_ROOT / "wav_converted"
    wav_dir.mkdir(exist_ok=True)
    
    # Load metadata
    print("\nLoading metadata...")
    train_df = pd.read_csv(TRAIN_METADATA, header=None, names=CSV_COLUMNS)
    test_df = pd.read_csv(TEST_METADATA, header=None, names=CSV_COLUMNS)
    
    # Filter for Minangkabau
    train_df = train_df[train_df["language_code"] == LANGUAGE]
    test_df = test_df[test_df["language_code"] == LANGUAGE]
    
    merged_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Total Minangkabau samples: {len(merged_df)}")
    
    # Convert each file
    print(f"\nConverting to WAV (16kHz) in: {wav_dir}")
    
    converted = 0
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Converting"):
        input_path = AUDIO_ROOT / row["audio_path"]
        
        # Create output path preserving structure
        relative_path = Path(row["audio_path"])
        output_path = wav_dir / relative_path.with_suffix(".wav")
        
        if convert_audio_to_wav(str(input_path), str(output_path)):
            converted += 1
    
    print(f"\n✅ Converted {converted}/{len(merged_df)} files to WAV")
    print(f"📁 Output directory: {wav_dir}")
    
    # Create new metadata file with WAV paths
    metadata_wav = AUDIO_ROOT / "metadata_minang_wav.csv"
    merged_df["wav_path"] = merged_df["audio_path"].apply(
        lambda x: str(Path("wav_converted") / Path(x).with_suffix(".wav"))
    )
    merged_df.to_csv(metadata_wav, index=False, header=False)
    print(f"📄 New metadata saved: {metadata_wav}")


if __name__ == "__main__":
    main()
