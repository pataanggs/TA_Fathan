"""
Text normalizer for post-processing.
Applies deterministic text normalization to BOTH predictions and references
before computing metrics. This ensures fair evaluation.
"""

import re
import string
from typing import Optional

from config import NORMALIZATION_CONFIG


def normalize_text(text: Optional[str]) -> str:
    """
    Apply full text normalization pipeline.

    Applied to BOTH predictions and references for fair comparison:
    1. Strip leading whitespace (Whisper adds leading space)
    2. Lowercase
    3. Remove punctuation
    4. Normalize whitespace

    Args:
        text: Raw text from model prediction or reference

    Returns:
        Normalized text
    """
    if text is None or not isinstance(text, str):
        return ""

    # Strip leading/trailing whitespace (Whisper often prepends a space)
    if NORMALIZATION_CONFIG["strip_leading_space"]:
        text = text.strip()

    # Lowercase
    if NORMALIZATION_CONFIG["lowercase"]:
        text = text.lower()

    # Remove punctuation
    if NORMALIZATION_CONFIG["remove_punctuation"]:
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Normalize whitespace
    if NORMALIZATION_CONFIG["normalize_whitespace"]:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_pair(prediction: str, reference: str) -> tuple:
    """
    Normalize both prediction and reference consistently.

    Args:
        prediction: Model prediction text
        reference: Ground truth reference text

    Returns:
        Tuple of (normalized_prediction, normalized_reference)
    """
    return normalize_text(prediction), normalize_text(reference)


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT NORMALIZER — Test Cases")
    print("=" * 60)

    test_cases = [
        (
            " Sesungguhnya sikap tak paduli, dan malah cekkan!",
            "sasungguahnyo sikap tak paduli dan malecehkan",
        ),
        (
            " sabagai cito-cito paliang tinggi",
            "sabagai cito cito paliang tinggi",
        ),
        (
            "  Bantulah  mau ujutkan  ",
            "bantulah mawujudkan",
        ),
    ]

    for pred, ref in test_cases:
        norm_pred, norm_ref = normalize_pair(pred, ref)
        print(f"\nPred raw:  {repr(pred)}")
        print(f"Pred norm: {repr(norm_pred)}")
        print(f"Ref  raw:  {repr(ref)}")
        print(f"Ref  norm: {repr(norm_ref)}")
