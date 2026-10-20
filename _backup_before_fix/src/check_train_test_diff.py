"""
Compares training data vs testing data to verify the inter-patient split
is working correctly, and reports statistical differences between the two.

Run this AFTER feature_extraction.py has created ecg_features.csv.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from scipy import stats
import os

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")
FEATURE_COLUMNS = ["rr_prev", "rr_next", "amplitude", "local_mean", "local_std"]


def main():
    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    groups = df["record"]

    # Recreate the same inter-patient split used in train_models.py
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # ---------------------------------------------------------------
    # CHECK 1: Confirm no patient overlap between train and test
    # ---------------------------------------------------------------
    train_patients = set(train_df["record"])
    test_patients = set(test_df["record"])
    overlap = train_patients & test_patients

    print("="*60)
    print("CHECK 1: Patient Overlap")
    print("="*60)
    print(f"Train patients: {sorted(train_patients)}")
    print(f"Test patients : {sorted(test_patients)}")
    print(f"Overlapping patients: {overlap if overlap else 'NONE (correct)'}")

    # ---------------------------------------------------------------
    # CHECK 2: Class balance comparison
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("CHECK 2: Class Balance (Normal=0 vs Abnormal=1)")
    print("="*60)
    train_balance = train_df["label"].value_counts(normalize=True).sort_index()
    test_balance = test_df["label"].value_counts(normalize=True).sort_index()

    balance_df = pd.DataFrame({
        "Train (%)": (train_balance * 100).round(2),
        "Test (%)": (test_balance * 100).round(2)
    })
    print(balance_df)

    # ---------------------------------------------------------------
    # CHECK 3: Feature distribution comparison (mean, std, and
    # statistical test for whether distributions differ significantly)
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("CHECK 3: Feature Distributions (Train vs Test)")
    print("="*60)

    summary_rows = []
    for col in FEATURE_COLUMNS:
        train_vals = train_df[col]
        test_vals = test_df[col]

        # Kolmogorov-Smirnov test: checks if two distributions differ
        # p < 0.05 suggests a statistically significant difference
        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)

        summary_rows.append({
            "feature": col,
            "train_mean": train_vals.mean(),
            "test_mean": test_vals.mean(),
            "train_std": train_vals.std(),
            "test_std": test_vals.std(),
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "significantly_different": "Yes" if p_value < 0.05 else "No"
        })

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))

    print("\nNote: A statistically significant difference (p < 0.05) between "
          "train and test feature distributions is EXPECTED and NORMAL in an "
          "inter-patient split, since different patients naturally have "
          "somewhat different heartbeat characteristics. This is not an error "
          "— it's part of what makes inter-patient evaluation a more "
          "realistic, harder test than a random split.")

    # Save for reference / inclusion in paper appendix if useful
    output_path = os.path.join(os.path.dirname(__file__), "..", "models",
                                "train_vs_test_distribution_check.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed comparison to {output_path}")


if __name__ == "__main__":
    main()
