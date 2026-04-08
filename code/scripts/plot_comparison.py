import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = BASE_DIR / "code"
OUTPUT_PATH = BASE_DIR / "figure" / "zero_shot_comparison.png"

ZERO_SHOT_WER = 89.44
ZERO_SHOT_CER = 31.02


def select_best_run(pattern: str):
    candidates = []
    for summary_path in (CODE_DIR.glob(pattern)):
        data = json.loads(summary_path.read_text())
        cv = data.get("cross_validation_summary", {})
        mean_wer = cv.get("mean_wer_percent")
        mean_cer = cv.get("mean_cer_percent")
        if mean_wer is None or mean_cer is None:
            continue
        candidates.append((mean_wer, mean_cer, summary_path.parent.name, cv))
    if not candidates:
        raise RuntimeError(f"No valid summary found for pattern: {pattern}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0]


fe_mean_wer, fe_mean_cer, fe_run_name, _ = select_best_run(
    "freeze_encoder/outputs/metrics/run_*/cross_validation_summary.json"
)
lora_mean_wer, lora_mean_cer, lora_run_name, _ = select_best_run(
    "lora/outputs/metrics/lora_*/cross_validation_summary.json"
)

models = [
    "Whisper Base\n(Zero-Shot)",
    f"Freeze Encoder\n({fe_run_name})",
    f"LoRA\n({lora_run_name})",
]
wer_values = [ZERO_SHOT_WER, fe_mean_wer, lora_mean_wer]
cer_values = [ZERO_SHOT_CER, fe_mean_cer, lora_mean_cer]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, wer_values, width, label="WER (%)", color="#ff9999", edgecolor="black", linewidth=1)
rects2 = ax.bar(x + width / 2, cer_values, width, label="CER (%)", color="#66b3ff", edgecolor="black", linewidth=1)

ax.set_ylabel("Persentase Error (%)", fontsize=12)
ax.set_title("Perbandingan Performa: Zero-Shot vs Best Freeze Encoder vs Best LoRA", fontsize=14, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.legend(fontsize=11)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)

ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300)

print("Best FE run:", fe_run_name, f"WER={fe_mean_wer:.2f}% CER={fe_mean_cer:.2f}%")
print("Best LoRA run:", lora_run_name, f"WER={lora_mean_wer:.2f}% CER={lora_mean_cer:.2f}%")
print(f"Plot saved to {OUTPUT_PATH}")