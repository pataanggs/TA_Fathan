import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Whisper Base\n(Zero-Shot)', 'Freeze Encoder\n(FE-9)', 'LoRA\n(L-10)']
wer_values = [89.44, 18.24, 15.67]
cer_values = [31.02, 3.87, 3.32]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars
rects1 = ax.bar(x - width/2, wer_values, width, label='WER (%)', color='#ff9999', edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, cer_values, width, label='CER (%)', color='#66b3ff', edgecolor='black', linewidth=1)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Persentase Error (%)', fontsize=12)
ax.set_title('Perbandingan Performa: Zero-Shot vs Freeze Encoder vs LoRA', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)

# Attach a text label above each bar in *rects*, displaying its height.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.ylim(0, 100) # Give some space for text
plt.grid(axis='y', linestyle='--', alpha=0.7)
fig.tight_layout()

# Save the plot
plt.savefig('figure/zero_shot_comparison.png', dpi=300)
print('Plot saved to figure/zero_shot_comparison.png')