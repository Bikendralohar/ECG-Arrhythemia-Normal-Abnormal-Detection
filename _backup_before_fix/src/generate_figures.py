"""
Generates publication-quality figures for your paper:
1. Bar chart comparing F1-scores: Intra-patient vs Inter-patient
2. Confusion matrix heatmap for the best inter-patient model (SVM)

Run this AFTER train_models.py has completed and saved
models/model_comparison_both_splits.csv

Output: high-resolution PNG files saved to models/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR = os.path.join(MODEL_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")
FEATURE_COLUMNS = ["rr_prev", "rr_next", "amplitude", "local_mean", "local_std"]

# Use a clean, publication-friendly plot style
plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_f1_comparison():
    """Figure 1: Bar chart of F1-scores, Intra-patient vs Inter-patient."""
    results_path = os.path.join(MODEL_DIR, "model_comparison_both_splits.csv")
    df = pd.read_csv(results_path)

    models = df["model"].unique()
    intra_f1 = df[df["evaluation_type"].str.contains("Intra")].set_index("model")["f1"]
    inter_f1 = df[df["evaluation_type"].str.contains("Inter")].set_index("model")["f1"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width/2, intra_f1.loc[models], width, label="Intra-patient", color="#4C72B0")
    bars2 = ax.bar(x + width/2, inter_f1.loc[models], width, label="Inter-patient", color="#DD8452")

    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Comparison: Intra-patient vs Inter-patient Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)

    output_path = os.path.join(FIG_DIR, "fig1_f1_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 1 to {output_path}")


def plot_confusion_matrix_for_best_model():
    """Figure 2: Confusion matrix heatmap for the saved best model (inter-patient)."""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["record"]

    # Recreate the exact same inter-patient split used during training
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=groups))
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Best Model (Inter-patient Evaluation)")

    output_path = os.path.join(FIG_DIR, "fig2_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Figure 2 to {output_path}")


if __name__ == "__main__":
    plot_f1_comparison()
    plot_confusion_matrix_for_best_model()
    print("\nBoth figures are ready to insert directly into your paper.")