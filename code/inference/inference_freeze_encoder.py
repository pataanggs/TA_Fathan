import os
import glob
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

def transcribe_audio(audio_path, model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    audio, rate = librosa.load(audio_path, sr=16000)

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(**inputs)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    del model
    torch.cuda.empty_cache()

    return transcription[0]

if __name__ == "__main__":
    # TODO: 1. Pilih fold dengan hasil evaluasi (WER/CER) terbaik dari saat training
    # TODO: 2. Copy path folder checkpoint dari fold terbaik tersebut ke sini
    BEST_CHECKPOINT_PATH = "../freeze_encoder/outputs/checkpoints/fold_0/checkpoint-84" # <-- Ganti jika fold lain lebih baik
    
    # TODO: 3. Masukkan path file audio yang ingin ditranskripsi saat demo
    AUDIO_FILE = "../Data/test/minangkabau/universal-declaration-of-human-rights/human_rights_un_min_sd_0075.mp3"

    print(f"=== DEMO SKRIPSI: Inferensi Model Freeze Encoder ===")
    print(f"File Audio  : {AUDIO_FILE}")
    print(f"Model Path  : {BEST_CHECKPOINT_PATH}\n")

    try:
        print("Memuat model dan melakukan transkripsi (harap tunggu)...")
        result = transcribe_audio(AUDIO_FILE, BEST_CHECKPOINT_PATH)
        print("\n" + "="*50)
        print("HASIL TRANSKRIPSI:")
        print("="*50)
        print(result)
        print("="*50 + "\n")
    except Exception as e:
        print(f"-> Terjadi error: {e}\n")
