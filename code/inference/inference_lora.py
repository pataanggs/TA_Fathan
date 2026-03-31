import os
import glob
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from peft import PeftModel, PeftConfig

def transcribe_audio_lora(audio_path, peft_model_id):
    # Load konfigurasi LoRA untuk mengetahui Model Dasarnya (Base Model)
    config = PeftConfig.from_pretrained(peft_model_id)
    
    base_model_name = config.base_model_name_or_path
    processor = AutoProcessor.from_pretrained(base_model_name)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model_name)
    
    # Menggabungkan model dasar dengan bobot adapter LoRA hasil fine-tuning
    model = PeftModel.from_pretrained(base_model, peft_model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Mengubah audio menjadi sampling rate 16kHz
    audio, rate = librosa.load(audio_path, sr=16000)

    # Preprocessing
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        predicted_ids = model.generate(**inputs)

    # Decode id ke dalam bentuk teks
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    # Bersihkan memory GPU (opsional tapi disarankan bila banyak model dipanggil bergantian)
    del model
    del base_model
    torch.cuda.empty_cache()

    return transcription[0]

if __name__ == "__main__":
    # Path relatif yang benar naik satu tingkat (ke folder `code`), lalu ke lora
    BEST_CHECKPOINT_PATH = "../lora/outputs/checkpoints/fold_0/checkpoint-132" 
    
    # Path file audio Anda
    AUDIO_FILE = "../Data/test/minangkabau/universal-declaration-of-human-rights/human_rights_un_min_sd_0075.mp3"

    print(f"=== DEMO SKRIPSI: Inferensi Model LoRA ===")
    print(f"File Audio  : {AUDIO_FILE}")
    print(f"Model Path  : {BEST_CHECKPOINT_PATH}\n")

    try:
        print("Memuat model dan melakukan transkripsi (harap tunggu)...")
        result = transcribe_audio_lora(AUDIO_FILE, BEST_CHECKPOINT_PATH)
        print("\n" + "="*50)
        print("HASIL TRANSKRIPSI:")
        print("="*50)
        print(result)
        print("="*50 + "\n")
    except Exception as e:
        print(f"-> Terjadi error: {e}\n")
