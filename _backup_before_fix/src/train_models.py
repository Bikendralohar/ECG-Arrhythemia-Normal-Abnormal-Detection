"""
Part 3: Train and compare classical ML models for abnormal heartbeat detection.

Supports TWO evaluation strategies, and reports both:

1. INTRA-PATIENT (the standard/naive split): beats are split randomly,
   so the same patient can appear in both train and test sets. This is
   what most quick tutorials do, and it tends to overestimate real-world
   performance because the model may partially "recognize" a patient's
   own heartbeat style from training.

2. INTER-PATIENT (the more rigorous split): entire patients are held out
   for testing, so the model NEVER sees a test patient's data during
   training. This is a stricter, more clinically realistic evaluation of
   whether the model generalizes to brand-new patients.

Run this AFTER feature_extraction.py has created ecg_features.csv.
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = ["rr_prev", "rr_next", "amplitude", "local_mean", "local_std"]


def load_dataset():
    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["record"]  # patient/record ID, needed for inter-patient split
    return X, y, groups


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "auroc": roc_auc_score(y_test, probs),
    }

    print(f"\n--- {name} ---")
    for k in ["accuracy", "precision", "recall", "f1", "auroc"]:
        print(f"{k.capitalize():10}: {result[k]:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return result


def train_and_evaluate_all(X_train, X_test, y_train, y_test, split_label):
    """Train all 4 models on a given train/test split and return results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION STRATEGY: {split_label}")
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"{'='*60}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    trained_models = {}

    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_scaled, y_test))
    trained_models["Logistic Regression"] = lr

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))
    trained_models["Random Forest"] = rf

    svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
    svm.fit(X_train_scaled, y_train)
    results.append(evaluate_model("SVM", svm, X_test_scaled, y_test))
    trained_models["SVM"] = svm

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=200, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )
        xgb.fit(X_train, y_train)
        results.append(evaluate_model("XGBoost", xgb, X_test, y_test))
        trained_models["XGBoost"] = xgb
    else:
        print("\n[Note] xgboost not installed — skipping XGBoost.")

    results_df = pd.DataFrame(results)
    results_df["evaluation_type"] = split_label
    return results_df, trained_models, scaler


def main():
    X, y, groups = load_dataset()
    print(f"Loaded dataset: {X.shape[0]} beats, {X.shape[1]} features")
    print("Label distribution:\n", y.value_counts())
    print(f"Number of unique patients/records: {groups.nunique()}")

    all_results = []

    # ---------------------------------------------------------------
    # STRATEGY 1: Intra-patient split (random split, standard/naive)
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    intra_results, intra_models, intra_scaler = train_and_evaluate_all(
        X_train, X_test, y_train, y_test, "Intra-patient (random split)"
    )
    all_results.append(intra_results)

    # ---------------------------------------------------------------
    # STRATEGY 2: Inter-patient split (entire patients held out)
    # ---------------------------------------------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train_ip, X_test_ip = X.iloc[train_idx], X.iloc[test_idx]
    y_train_ip, y_test_ip = y.iloc[train_idx], y.iloc[test_idx]

    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])
    print(f"\nInter-patient split -> Train patients: {len(train_patients)}, "
          f"Test patients: {len(test_patients)} (no overlap: "
          f"{len(train_patients & test_patients) == 0})")

    inter_results, inter_models, inter_scaler = train_and_evaluate_all(
        X_train_ip, X_test_ip, y_train_ip, y_test_ip, "Inter-patient (patient-held-out)"
    )
    all_results.append(inter_results)

    # ---------------------------------------------------------------
    # Combine both result sets into one comparison table
    # ---------------------------------------------------------------
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df[["evaluation_type", "model", "accuracy",
                                "precision", "recall", "f1", "auroc"]]

    print("\n" + "="*60)
    print("FINAL COMPARISON: Intra-patient vs Inter-patient")
    print("="*60)
    print(combined_df.to_string(index=False))

    combined_df.to_csv(os.path.join(MODEL_DIR, "model_comparison_both_splits.csv"), index=False)
    print(f"\nSaved full comparison table to models/model_comparison_both_splits.csv")
    print("Use this table directly in your paper's Results section — "
          "it shows both the standard evaluation and the more rigorous "
          "inter-patient evaluation side by side.")

    # ---------------------------------------------------------------
    # Save the best model from the INTER-PATIENT results
    # (this is the more honest, deployment-realistic choice)
    # ---------------------------------------------------------------
    best_row = inter_results.loc[inter_results["f1"].idxmax()]
    best_name = best_row["model"]
    best_model = inter_models[best_name]

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(inter_scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    print(f"\nBest model based on inter-patient evaluation ({best_name}) "
          f"saved to models/best_model.pkl")


if __name__ == "__main__":
    main()