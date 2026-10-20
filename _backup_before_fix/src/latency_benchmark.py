"""
Latency Benchmark: Measures how fast the trained model can classify
a single heartbeat, and how many heartbeats per second it can process.

This provides real evidence for your paper's "real-time" claim, instead
of just calling the system real-time without measurements.

Run this AFTER train_models.py has created models/best_model.pkl
and models/scaler.pkl.
"""

import pandas as pd
import numpy as np
import joblib
import time
import os

FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURE_COLUMNS = ["rr_prev", "rr_next", "amplitude", "local_mean", "local_std"]

N_SINGLE_PREDICTIONS = 1000   # how many single, one-at-a-time predictions to time
N_BATCH_PREDICTIONS = 10000   # how many predictions to use for throughput test


def load_model_and_data():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    df = pd.read_csv(FEATURES_CSV)
    X = df[FEATURE_COLUMNS].values
    return model, scaler, X


def benchmark_single_prediction_latency(model, scaler, X, n=N_SINGLE_PREDICTIONS):
    """
    Simulates the real-time scenario: one heartbeat arrives, gets scaled,
    and gets scored — one at a time, exactly like the FastAPI endpoint does.
    """
    sample_indices = np.random.choice(len(X), size=n, replace=False)
    latencies_ms = []

    # Warm-up run (first call is often slower due to internal setup) — not counted
    _ = model.predict(scaler.transform(X[sample_indices[:1]]))

    for idx in sample_indices:
        row = X[idx].reshape(1, -1)

        start = time.perf_counter()
        row_scaled = scaler.transform(row)
        _ = model.predict_proba(row_scaled)
        end = time.perf_counter()

        latencies_ms.append((end - start) * 1000)  # convert to milliseconds

    return np.array(latencies_ms)


def benchmark_batch_throughput(model, scaler, X, n=N_BATCH_PREDICTIONS):
    """
    Measures how many heartbeats per second the model can process
    when scoring many at once (useful for showing scalability).
    """
    n = min(n, len(X))
    batch = X[:n]

    start = time.perf_counter()
    batch_scaled = scaler.transform(batch)
    _ = model.predict_proba(batch_scaled)
    end = time.perf_counter()

    total_time = end - start
    throughput = n / total_time  # predictions per second
    return total_time, throughput


def main():
    print("Loading model and data...")
    model, scaler, X = load_model_and_data()
    print(f"Loaded {len(X)} heartbeat samples for benchmarking.\n")

    print(f"Running single-prediction latency test ({N_SINGLE_PREDICTIONS} runs)...")
    latencies = benchmark_single_prediction_latency(model, scaler, X)

    print("\n=== Single-Prediction Latency (real-time, one beat at a time) ===")
    print(f"Mean latency   : {latencies.mean():.4f} ms")
    print(f"Median latency : {np.median(latencies):.4f} ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.4f} ms")
    print(f"Max latency    : {latencies.max():.4f} ms")
    print(f"Min latency    : {latencies.min():.4f} ms")

    print(f"\nRunning batch throughput test ({N_BATCH_PREDICTIONS} predictions)...")
    total_time, throughput = benchmark_batch_throughput(model, scaler, X)

    print("\n=== Batch Throughput ===")
    print(f"Total time for {min(N_BATCH_PREDICTIONS, len(X))} predictions: {total_time:.4f} sec")
    print(f"Throughput: {throughput:.1f} predictions/second")

    # Save results to a small CSV for easy reference in your paper
    summary = pd.DataFrame([{
        "mean_latency_ms": latencies.mean(),
        "median_latency_ms": np.median(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "max_latency_ms": latencies.max(),
        "throughput_per_sec": throughput
    }])
    summary_path = os.path.join(MODEL_DIR, "latency_benchmark.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved benchmark summary to {summary_path}")

    print("\n=== Suggested sentence for your paper ===")
    print(f'"The proposed system achieves a mean inference latency of '
          f'{latencies.mean():.2f} ms per heartbeat (95th percentile: '
          f'{np.percentile(latencies, 95):.2f} ms), demonstrating suitability '
          f'for real-time clinical monitoring, with a measured throughput of '
          f'{throughput:.0f} predictions per second."')


if __name__ == "__main__":
    main()