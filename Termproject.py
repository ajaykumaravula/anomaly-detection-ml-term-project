# ================================
# Term Project - Anomaly Detection
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow.keras import layers, models

# ====================================================
# 1. LOAD DATASET
# ====================================================
df = pd.read_csv("creditcard.csv")   # Place file in the same folder
print(df.head())
print(df["Class"].value_counts())

# ====================================================
# 2. CLASS DISTRIBUTION GRAPH
# ====================================================
plt.figure(figsize=(6,4))
sns.countplot(x=df['Class'])
plt.title("Class Distribution (0=Normal, 1=Fraud)")
plt.show()

# ====================================================
# 3. DATA PREPROCESSING
# ====================================================
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train only on normal transactions
X_train = X_scaled[y==0]
X_test = X_scaled
y_test = y

# ====================================================
# 4. ISOLATION FOREST
# ====================================================
iso = IsolationForest(contamination=0.001, random_state=42)
iso.fit(X_train)

y_pred_iso = iso.predict(X_test)
y_pred_iso = [1 if x==-1 else 0 for x in y_pred_iso]

# Metrics
iso_precision = precision_score(y_test, y_pred_iso)
iso_recall = recall_score(y_test, y_pred_iso)
iso_f1 = f1_score(y_test, y_pred_iso)

print("\n=== Isolation Forest ===")
print(confusion_matrix(y_test, y_pred_iso))
print(classification_report(y_test, y_pred_iso))

# ====================================================
# 5. ONE-CLASS SVM (using smaller subset for speed)
# ====================================================
subset_size = 20000
X_train_svm = X_train[:subset_size]

ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.001)
ocsvm.fit(X_train_svm)

y_pred_svm = ocsvm.predict(X_test)
y_pred_svm = [1 if x==-1 else 0 for x in y_pred_svm]

# Metrics
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)

print("\n=== One-Class SVM ===")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ====================================================
# 6. AUTOENCODER
# ====================================================
input_dim = X_train.shape[1]

autoencoder = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(input_dim,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)

# Reconstruction error
X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
threshold = np.percentile(mse, 99)
y_pred_ae = (mse > threshold).astype(int)

# Metrics
ae_precision = precision_score(y_test, y_pred_ae)
ae_recall = recall_score(y_test, y_pred_ae)
ae_f1 = f1_score(y_test, y_pred_ae)

print("\n=== Autoencoder ===")
print(confusion_matrix(y_test, y_pred_ae))
print(classification_report(y_test, y_pred_ae))

# ====================================================
# 7. COMPARISON GRAPH (All Three Methods)
# ====================================================
methods = ['Isolation Forest', 'One-Class SVM', 'Autoencoder']
precision = [iso_precision, svm_precision, ae_precision]
recall = [iso_recall, svm_recall, ae_recall]
f1_scores = [iso_f1, svm_f1, ae_f1]

x = np.arange(len(methods))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision, width, label='Precision', color='skyblue')
plt.bar(x, recall, width, label='Recall', color='orange')
plt.bar(x + width, f1_scores, width, label='F1-score', color='green')

plt.xticks(x, methods)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Comparison of Anomaly Detection Methods")
plt.legend()

# Add values on top of bars
for i in range(len(methods)):
    plt.text(i - width, precision[i]+0.02, f"{precision[i]:.2f}")
    plt.text(i, recall[i]+0.02, f"{recall[i]:.2f}")
    plt.text(i + width, f1_scores[i]+0.02, f"{f1_scores[i]:.2f}")

plt.show()
