"""
Data augmentation module for Whisper fine-tuning.
Implements speed perturbation, noise injection, and SpecAugment.
Applied to training data only.
"""

import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Optional, Tuple

from config import SAMPLE_RATE, AUGMENTATION_CONFIG


class SpeedPerturbation:
    """Apply speed perturbation to audio."""
    
    def __init__(self, speeds: list = None):
        """
        Args:
            speeds: List of speed factors (default: [0.9, 1.0, 1.1])
        """
        self.speeds = speeds or AUGMENTATION_CONFIG["speed_perturbation"]
    
    def __call__(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Apply random speed perturbation.
        
        Args:
            audio: Input audio array
            sample_rate: Sample rate
            
        Returns:
            Speed-perturbed audio
        """
        speed_factor = random.choice(self.speeds)
        
        if speed_factor == 1.0:
            return audio
        
        # Convert to tensor for torchaudio processing
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        
        # Create resampler for speed change
        # Speed up = higher resample rate, Speed down = lower resample rate
        new_sample_rate = int(sample_rate * speed_factor)
        resampler = T.Resample(orig_freq=new_sample_rate, new_freq=sample_rate)
        
        # First, change the playback rate by resampling
        effects = [["speed", str(speed_factor)], ["rate", str(sample_rate)]]
        
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio_tensor, sample_rate, effects
            )
            return augmented.squeeze(0).numpy()
        except:
            # Fallback: simple resampling
            return audio


class NoiseInjection:
    """Add random noise to audio."""
    
    def __init__(self, snr_range: Tuple[int, int] = None):
        """
        Args:
            snr_range: Range of SNR values in dB (default: (15, 25))
        """
        self.snr_range = snr_range or AUGMENTATION_CONFIG["noise_snr_range"]
    
    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise with random SNR.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio with added noise
        """
        snr_db = random.uniform(*self.snr_range)
        
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Calculate noise power based on SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        
        # Add noise to signal
        augmented = audio + noise
        
        return augmented.astype(np.float32)


class SpecAugment:
    """Apply SpecAugment to mel spectrogram."""
    
    def __init__(
        self, 
        time_mask_param: int = None, 
        freq_mask_param: int = None
    ):
        """
        Args:
            time_mask_param: Maximum time mask length (T parameter)
            freq_mask_param: Maximum frequency mask length (F parameter)
        """
        self.time_mask_param = time_mask_param or AUGMENTATION_CONFIG["specaugment_time_mask"]
        self.freq_mask_param = freq_mask_param or AUGMENTATION_CONFIG["specaugment_freq_mask"]
        
        self.time_masking = T.TimeMasking(time_mask_param=self.time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)
    
    def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time and frequency masking to mel spectrogram.
        
        Args:
            mel_spectrogram: Input mel spectrogram tensor
            
        Returns:
            Augmented mel spectrogram
        """
        # Apply time masking
        augmented = self.time_masking(mel_spectrogram)
        
        # Apply frequency masking
        augmented = self.freq_masking(augmented)
        
        return augmented


class AudioAugmenter:
    """Combined audio augmentation pipeline for training."""
    
    def __init__(
        self,
        use_speed_perturbation: bool = True,
        use_noise_injection: bool = True,
        use_specaugment: bool = True,
        augmentation_prob: float = 0.8  # Increased for better regularization
    ):
        """
        Args:
            use_speed_perturbation: Whether to use speed perturbation
            use_noise_injection: Whether to use noise injection
            use_specaugment: Whether to use SpecAugment
            augmentation_prob: Probability of applying each augmentation
        """
        self.augmentation_prob = augmentation_prob
        
        self.speed_perturbation = SpeedPerturbation() if use_speed_perturbation else None
        self.noise_injection = NoiseInjection() if use_noise_injection else None
        self.specaugment = SpecAugment() if use_specaugment else None
    
    def augment_waveform(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Apply waveform-level augmentations.
        
        Args:
            audio: Input audio waveform
            sample_rate: Sample rate
            
        Returns:
            Augmented audio waveform
        """
        # Speed perturbation
        if self.speed_perturbation and random.random() < self.augmentation_prob:
            audio = self.speed_perturbation(audio, sample_rate)
        
        # Noise injection
        if self.noise_injection and random.random() < self.augmentation_prob:
            audio = self.noise_injection(audio)
        
        return audio
    
    def augment_spectrogram(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply spectrogram-level augmentations (SpecAugment).
        
        Args:
            mel_spectrogram: Input mel spectrogram
            
        Returns:
            Augmented mel spectrogram
        """
        if self.specaugment and random.random() < self.augmentation_prob:
            mel_spectrogram = self.specaugment(mel_spectrogram)
        
        return mel_spectrogram


if __name__ == "__main__":
    # Test augmentation
    print("Testing Audio Augmentation Module")
    print("-" * 50)
    
    # Create test audio (1 second of sine wave)
    duration = 1.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    print(f"Original audio shape: {test_audio.shape}")
    print(f"Original audio range: [{test_audio.min():.4f}, {test_audio.max():.4f}]")
    
    # Test augmenter
    augmenter = AudioAugmenter()
    augmented = augmenter.augment_waveform(test_audio)
    
    print(f"\nAugmented audio shape: {augmented.shape}")
    print(f"Augmented audio range: [{augmented.min():.4f}, {augmented.max():.4f}]")
