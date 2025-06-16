import matplotlib.pyplot as plt

# asr = [39.1, 28.3, 75.5, 25.5, 20.7]
models = ['LLaMA3', 'Phi4', 'Mistral-7B',  'DeepSeek-R1', 'Gemma3']
garak = [39.1, 33.40, 47.40,  32.8, 87.40]
redhit = [59.80, 37.75, 49.05,  48.40, 75.20]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, redhit, color='skyblue', edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1,
             f'{yval:.1f}%', ha='center', va='bottom')

plt.ylabel('Attack Success Rate (%)')
plt.title('RedHit\'s ASR Across Different Models')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
c = 1
