# Whisper Fine-Tuning for Minangkabau ASR

Fine-tuning OpenAI Whisper Base (74M params) for Minangkabau speech recognition using two strategies with 5-fold cross-validation.

## Project Structure

```
code/
├── freeze_encoder/          # Strategy 1: Freeze Encoder (Decoder-Only)
│   ├── config.py            # Hyperparameters & paths
│   ├── trainer.py           # Model init, freeze encoder, training loop
│   ├── main.py              # Entry point (5-fold CV)
│   └── outputs/             # Results & metrics
│
├── lora/                    # Strategy 2: LoRA (PEFT)
│   ├── config.py            # LoRA hyperparameters & paths
│   ├── trainer.py           # Model init, LoRA adapters, training loop
│   ├── main.py              # Entry point (5-fold CV)
│   ├── requirements.txt     # Additional dep: peft>=0.13.0
│   └── outputs/             # Results & metrics
│
├── data_loader.py           # [SHARED] Dataset loading & merging
├── dataset.py               # [SHARED] HF dataset preparation & collation
├── augmentation.py          # [SHARED] Audio augmentation pipeline
├── audio_preprocessing.py   # [SHARED] Audio loading & normalization
├── text_preprocessing.py    # [SHARED] Text cleaning functions
├── metrics_logger.py        # [SHARED] Comprehensive metrics logging
├── convert_audio.py         # [UTILITY] MP3-to-WAV conversion
├── visualize.py             # [UTILITY] Training visualization
├── visualize_metrics.py     # [UTILITY] Metrics dashboard
├── requirements.txt         # Base Python dependencies
├── .env                     # API keys (WANDB_API_KEY)
└── Data/                    # Audio files & metadata CSVs
```

## Strategy Comparison

| Aspect | Freeze Encoder | LoRA (PEFT) |
|---|---|---|
| Trainable Params | ~37M (50%) | ~197K (0.27%) |
| Learning Rate | 1e-5 | 5e-4 |
| Max Steps | 200 | 400 |
| Weight Decay | 0.2 | 0.01 |
| Dropout | 0.3/0.2/0.2 | 0.1/0.05/0.05 |
| Early Stop Patience | 8 steps | 10 steps |
| Regularization | High dropout + weight decay | Implicit (low-rank constraint) |

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
# For LoRA strategy, also install:
pip install -r lora/requirements.txt
```

### Freeze Encoder Strategy
```bash
cd code/freeze_encoder
python main.py
```

### LoRA Strategy
```bash
cd code/lora
python main.py
```

## How It Works

Both strategies share the same:
- Data loading pipeline (`data_loader.py`)
- Audio augmentation (`augmentation.py`)
- Dataset preparation (`dataset.py`)
- Metrics logging (`metrics_logger.py`)
- 5-fold cross-validation framework

Each strategy's `main.py` registers its own `config.py` as the `config` module before importing shared modules. This ensures shared code (which does `from config import SAMPLE_RATE, ...`) uses the correct strategy-specific settings.

## Dataset

- **Language**: Minangkabau (min) with Indonesian (id) as proxy language token
- **Size**: 156 audio files (~1.3 hours)
- **Format**: WAV, 16kHz mono
- **Evaluation**: 5-fold cross-validation (no held-out test set)
