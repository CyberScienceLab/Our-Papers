
import random
import matplotlib.pyplot as plt
import os


plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['legend.title_fontsize'] = 15

llm_names = ['LLaMA3', 'Phi-4', 'Mistral', 'DeepSeek-R1', 'Gemma-3']

model_names = {ix: name for ix, name in enumerate(llm_names)}

accuracies = {name: [] for name in llm_names}

current_path = os.path.abspath(__file__)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
accuracies_in_treshold = {name: [] for name in llm_names}

for name in llm_names:

    with open(os.path.join(current_path, '..', name+'.txt'), "r") as file:
        for line in file:
            try:
                acc = float(line.strip())
                accuracies[name].append(acc)
            except Exception as e:
                print(f'error: {e}')

    for treshold in thresholds:
        accuracies_in_treshold[name].append(len([
            acc for acc in accuracies[name] if acc >= treshold]))


plt.figure(figsize=(8, 6))
i = 0
for model, counts in accuracies_in_treshold.items():
    plt.plot(thresholds, counts, marker='o', linestyle=[
             ':', '-', '-.', '--', '-'][i], label=model)
    i += 1

plt.title("Accuracy ≥ Threshold Counts per LLM")
plt.xlabel("Threshold")
plt.ylabel("Number of Predictions ≥ Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
c = 1
