import matplotlib.pyplot as plt
import numpy as np
import os

# UPDATE THESE NUMBERS AFTER RUNNING EVAL
results = {
    "Standard SFT": {"Pass Rate": 0, "Correction Rate": 0},
    "Trace Distillation": {"Pass Rate": 0, "Correction Rate": 0}
}

def plot():
    os.makedirs("./plots", exist_ok=True)
    methods = list(results.keys())
    pass_rates = [results[m]["Pass Rate"] for m in methods]
    correction_rates = [results[m]["Correction Rate"] for m in methods]
    
    x = np.arange(len(methods))
    
    fig, ax = plt.subplots(figsize=(8,6))
    bars = ax.bar(x, pass_rates, 0.4, color=['red', 'blue'])
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel('Success Rate (%)'); ax.set_title('Zero-Shot Tool Generalization')
    plt.savefig("./plots/results.png")
    print("Plot saved to ./plots/results.png")

if __name__ == "__main__":
    plot()