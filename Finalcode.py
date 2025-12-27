import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['Z-score', 'Isolation Forest', 'One-Class SVM', 'Autoencoder']
auc_scores = [0.885497, 0.639484, 0.500017, 0.706188]

# Create DataFrame (optional, for table use)
df_auc = pd.DataFrame({
    "Model": models,
    "AUC ROC": auc_scores
})
print(df_auc)

# Plot
plt.figure(figsize=(6,4))
bars = plt.bar(models, auc_scores)

plt.ylabel("AUC-ROC")
plt.ylim(0,1)
plt.title("Anomaly Detection Model Comparison")

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + 0.02,
             f"{height:.2f}",
             ha='center',
             fontsize=9)

plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
