"""
Dataset module for Whisper fine-tuning.
Implements HuggingFace-compatible Dataset and DataCollator.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from datasets import Dataset
from transformers import WhisperProcessor

from config import SAMPLE_RATE, LANGUAGE, TASK
from audio_preprocessing import load_processed_audio
from text_preprocessing import preprocess_text
from augmentation import AudioAugmenter


def prepare_dataset(
    batch: Dict,
    processor: WhisperProcessor,
    augmenter: Optional[AudioAugmenter] = None,
    is_training: bool = True
) -> Dict:
    """
    Prepare a single batch for Whisper model.
    
    Args:
        batch: Dictionary containing 'audio_path' and 'text'
        processor: WhisperProcessor for tokenization
        augmenter: Optional AudioAugmenter for training augmentation
        is_training: Whether this is training data
        
    Returns:
        Dictionary with 'input_features' and 'labels'
    """
    # Load audio
    audio = load_processed_audio(batch["audio_path"])
    
    # Apply augmentation for training
    if is_training and augmenter is not None:
        audio = augmenter.augment_waveform(audio)
    
    # Get input features (mel spectrogram)
    input_features = processor.feature_extractor(
        audio, 
        sampling_rate=SAMPLE_RATE,
        return_tensors="np"
    ).input_features[0]
    
    # Apply SpecAugment for training
    if is_training and augmenter is not None:
        input_features_tensor = torch.from_numpy(input_features).unsqueeze(0)
        input_features_tensor = augmenter.augment_spectrogram(input_features_tensor)
        input_features = input_features_tensor.squeeze(0).numpy()
    
    batch["input_features"] = input_features
    
    # Preprocess and tokenize text
    text = preprocess_text(batch["text"])
    batch["labels"] = processor.tokenizer(text).input_ids
    
    return batch


def prepare_dataset_for_training(
    dataset: Dataset,
    processor: WhisperProcessor,
    augmenter: Optional[AudioAugmenter] = None
) -> Dataset:
    """
    Prepare entire dataset for training.
    
    Args:
        dataset: HuggingFace Dataset
        processor: WhisperProcessor
        augmenter: Optional AudioAugmenter
        
    Returns:
        Processed Dataset
    """
    def process_fn(batch):
        return prepare_dataset(batch, processor, augmenter, is_training=True)
    
    dataset = dataset.map(
        process_fn,
        remove_columns=dataset.column_names,
        desc="Preparing training dataset"
    )
    
    return dataset


def prepare_dataset_for_evaluation(
    dataset: Dataset,
    processor: WhisperProcessor
) -> Dataset:
    """
    Prepare dataset for evaluation (no augmentation).
    
    Args:
        dataset: HuggingFace Dataset
        processor: WhisperProcessor
        
    Returns:
        Processed Dataset
    """
    def process_fn(batch):
        return prepare_dataset(batch, processor, augmenter=None, is_training=False)
    
    dataset = dataset.map(
        process_fn,
        remove_columns=dataset.column_names,
        desc="Preparing evaluation dataset"
    )
    
    return dataset


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs and labels.
    Uses manual padding for labels to avoid WhisperTokenizerFast warnings.
    """
    processor: Any
    decoder_start_token_id: int
    
    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of features.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched and padded features
        """
        # Split inputs and labels since they need different padding
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        
        # Pad input features using feature_extractor
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        
        # Manual padding for labels to avoid tokenizer.pad() warning
        # Get all label sequences
        label_sequences = [feature["labels"] for feature in features]
        
        # Find max length
        max_label_len = max(len(seq) for seq in label_sequences)
        
        # Pad each sequence manually with pad_token_id
        pad_token_id = self.processor.tokenizer.pad_token_id
        padded_labels = []
        attention_masks = []
        
        for seq in label_sequences:
            seq_len = len(seq)
            padding_len = max_label_len - seq_len
            
            # Pad sequence
            if isinstance(seq, list):
                padded_seq = seq + [pad_token_id] * padding_len
            else:
                padded_seq = list(seq) + [pad_token_id] * padding_len
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * seq_len + [0] * padding_len
            
            padded_labels.append(padded_seq)
            attention_masks.append(mask)
        
        # Convert to tensors
        labels = torch.tensor(padded_labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        
        # Replace padding with -100 to ignore in loss
        labels = labels.masked_fill(attention_mask.ne(1), -100)
        
        # Remove BOS token if it was added
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


def create_data_collator(processor: WhisperProcessor, model) -> DataCollatorSpeechSeq2SeqWithPadding:
    """
    Create data collator for Whisper training.
    
    Args:
        processor: WhisperProcessor
        model: Whisper model (to get decoder_start_token_id)
        
    Returns:
        DataCollatorSpeechSeq2SeqWithPadding instance
    """
    return DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


if __name__ == "__main__":
    from transformers import WhisperProcessor
    from data_loader import load_and_merge_datasets, dataframe_to_hf_dataset
    
    print("Testing Dataset Module")
    print("-" * 50)
    
    # Load processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    # Load sample data
    df = load_and_merge_datasets()
    dataset = dataframe_to_hf_dataset(df)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    print(f"\nSample entry:")
    print(dataset[0])
