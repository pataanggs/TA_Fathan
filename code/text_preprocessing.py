"""
Text preprocessing module for Whisper fine-tuning.
Handles lowercasing, punctuation removal, and digit preservation.
"""

import re
import string
from typing import Optional


def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        Lowercased text
    """
    return text.lower()


def remove_punctuation(text: str, keep_digits: bool = True) -> str:
    """
    Remove punctuation from text while optionally keeping digits.
    
    Args:
        text: Input text
        keep_digits: If True, preserve numeric digits (default: True)
        
    Returns:
        Text with punctuation removed
    """
    if keep_digits:
        # Remove punctuation but keep digits and letters
        pattern = f"[{re.escape(string.punctuation)}]"
        text = re.sub(pattern, " ", text)
    else:
        # Remove all punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces and strip.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_text(text: Optional[str]) -> str:
    """
    Full text preprocessing pipeline.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation (keeping digits)
    3. Normalize whitespace
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Apply preprocessing steps
    text = lowercase_text(text)
    text = remove_punctuation(text, keep_digits=True)
    text = normalize_whitespace(text)
    
    return text


def batch_preprocess_texts(texts: list) -> list:
    """
    Preprocess a batch of texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of preprocessed texts
    """
    return [preprocess_text(text) for text in texts]


if __name__ == "__main__":
    # Test text preprocessing
    test_texts = [
        "Katahui bana hak hak awak ko!",
        "Pasal 15: Tiok urang punyo hak...",
        "Indak ado pambedaan,   umpamonyo   pambedaan ras",
        None,
        "",
    ]
    
    print("Text Preprocessing Test:")
    print("-" * 50)
    for text in test_texts:
        processed = preprocess_text(text)
        print(f"Original: {repr(text)}")
        print(f"Processed: {repr(processed)}")
        print()
