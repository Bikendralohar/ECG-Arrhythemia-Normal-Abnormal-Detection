"""
Part 2: Extract features from ECG beats across all downloaded records.

Produces a single CSV file where each row = one heartbeat,
with engineered features + a normal/abnormal label.

Run this AFTER load_data.py has downloaded the records you need.
"""

import wfdb
import numpy as np
import pandas as pd
import os

from load_data import DATA_DIR, ALL_RECORDS  # use full dataset

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")

# Beat labels considered "normal" in MIT-BIH annotations.
# Everything else is treated as "abnormal" for a simple binary classifier.
NORMAL_LABELS = {'N', 'L', 'R', 'e', 'j'}

WINDOW = 90  # number of samples to look at before/after each beat for shape stats


def extract_features_for_record(record_name, data_dir=DATA_DIR):
    """Extract one row of features per heartbeat in a single record."""
    record_path = os.path.join(data_dir, record_name)

    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, 0]          # use the first ECG channel
    fs = record.fs

    annotation = wfdb.rdann(record_path, 'atr')
    beat_samples = annotation.sample
    beat_symbols = annotation.symbol

    rows = []
    for i in range(1, len(beat_samples) - 1):
        idx = beat_samples[i]

        # Skip beats too close to the start/end of the signal to get a full window
        if idx - WINDOW < 0 or idx + WINDOW >= len(signal):
            continue

        # --- Feature 1 & 2: RR intervals (gap to previous/next beat) ---
        rr_prev = (beat_samples[i] - beat_samples[i - 1]) / fs
        rr_next = (beat_samples[i + 1] - beat_samples[i]) / fs

        # --- Feature 3: peak amplitude at the beat location ---
        amplitude = signal[idx]

        # --- Feature 4 & 5: local signal statistics around the beat ---
        window_signal = signal[idx - WINDOW: idx + WINDOW]
        local_mean = np.mean(window_signal)
        local_std = np.std(window_signal)

        # --- Label: normal (0) vs abnormal (1) ---
        symbol = beat_symbols[i]
        label = 0 if symbol in NORMAL_LABELS else 1

        rows.append({
            "record": record_name,
            "sample_index": idx,
            "rr_prev": rr_prev,
            "rr_next": rr_next,
            "amplitude": amplitude,
            "local_mean": local_mean,
            "local_std": local_std,
            "symbol": symbol,
            "label": label
        })

    return pd.DataFrame(rows)


def build_full_dataset(record_list=ALL_RECORDS):
    """Loop through all records and combine their features into one table."""
    all_dfs = []
    for record_name in record_list:
        print(f"Extracting features for record {record_name} ...")
        try:
            df = extract_features_for_record(record_name)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Skipped {record_name} due to error: {e}")

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


if __name__ == "__main__":
    dataset = build_full_dataset(ALL_RECORDS)

    print(f"\nTotal beats extracted: {len(dataset)}")
    print("Label distribution:")
    print(dataset["label"].value_counts())

    dataset.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved feature dataset to {OUTPUT_CSV}")