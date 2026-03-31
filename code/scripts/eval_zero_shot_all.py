import pandas as pd
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from evaluate import load
from tqdm import tqdm
import re
import os

print("Starting...")
processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="Indonesian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
if torch.cuda.is_available(): model = model.to("cuda")

df = pd.read_csv("code/Data/metadata_minang_wav.csv", header=None)
df.columns = ["mp3_path", "lang", "speaker", "text", "wav_path"]

wer_metric = load("wer")
cer_metric = load("cer")

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

refs, preds, raw_refs = [], [], []
for _, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = os.path.join("code/Data", row["wav_path"])
    audio, sr = librosa.load(wav_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    if torch.cuda.is_available(): input_features = input_features.to("cuda")
    with torch.no_grad(): predicted_ids = model.generate(input_features, language="id", task="transcribe")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    raw_refs.append(row["text"])
    refs.append(normalize_text(row["text"]))
    preds.append(normalize_text(transcription))

wer = wer_metric.compute(references=refs, predictions=preds)
cer = cer_metric.compute(references=refs, predictions=preds)
print(f"WER: {wer*100:.2f}%, CER: {cer*100:.2f}%")

pd.DataFrame({
    "Sampel": [f"Sampel {i+1}" for i in range(len(df))],
    "Referen (Truth)": refs,
    "Prediksi (Zero-Shot)": preds
}).to_csv("code/zero_shot_all.csv", index=False)
