# Whisper LoRA Fine-tuning for Minangkabau Language

This directory contains the LoRA (Low-Rank Adaptation) fine-tuning implementation
for Whisper Base on Minangkabau language, as a separate experiment from the
Freeze Encoder approach.

## Directory Structure

```
lora/
├── config.py          # LoRA-specific configuration (hyperparameters, PEFT settings)
├── trainer.py         # LoRA trainer with PEFT model initialization
├── main.py            # Main entry point for 5-fold cross-validation
├── requirements.txt   # Additional dependencies (peft library)
├── README.md          # This file
└── outputs/           # Training outputs (created during training)
    ├── checkpoints/   # Model checkpoints
    └── metrics/       # Training metrics and results
```

## Shared Modules

The following modules are reused from the parent `code/` directory:
- `data_loader.py` - Dataset loading and merging
- `dataset.py` - HuggingFace dataset preparation and data collator
- `augmentation.py` - Audio augmentation (SpecAugment, Speed, Noise)
- `metrics_logger.py` - Comprehensive metrics logging (local + WandB)
- `audio_preprocessing.py` - Audio loading and preprocessing
- `text_preprocessing.py` - Text normalization

## Key Differences from Freeze Encoder

| Parameter | Freeze Encoder | LoRA |
|-----------|---------------|------|
| Strategy | Freeze encoder, train full decoder | LoRA adapters on attention layers |
| Trainable Params | ~37M (50%) | ~197K (0.27%) |
| Learning Rate | 1e-5 | 5e-4 |
| Weight Decay | 0.2 | 0.01 |
| Dropout | 0.3 / 0.2 / 0.2 | 0.1 / 0.05 / 0.05 |
| Max Steps | 200 | 400 |
| Early Stopping | Patience 8 | Patience 10 |
| LoRA Rank | - | 8 |
| LoRA Alpha | - | 16 |
| Label Smoothing | No | 0.1 |

## Usage

```bash
# 1. Install base requirements (from parent directory)
cd /path/to/code
pip install -r requirements.txt

# 2. Install LoRA-specific requirements
cd lora
pip install -r requirements.txt

# 3. Run training
python main.py
```

## Configuration

Edit `config.py` to adjust:
- **LoRA parameters**: rank, alpha, dropout, target modules
- **Training parameters**: learning rate, batch size, max steps
- **Augmentation**: same as Freeze Encoder for fair comparison
- **Early stopping**: patience and threshold
