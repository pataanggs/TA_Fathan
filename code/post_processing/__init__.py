"""
Post-Processing Module for Minangkabau ASR.

Applies dictionary-based spell correction using Levenshtein distance
to improve WER/CER after model inference.

Modules:
  - config.py:             Paths, hyperparameters, and configuration
  - dictionary_builder.py: Build per-fold dictionaries from training transcripts
  - text_normalizer.py:    Text normalization (lowercase, punctuation, whitespace)
  - spell_corrector.py:    Levenshtein distance-based word correction
  - evaluator.py:          Full evaluation with WER/CER recomputation
  - main.py:               Pipeline orchestrator
"""
