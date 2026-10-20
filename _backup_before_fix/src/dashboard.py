"""
Part 5: Streamlit dashboard — simulates a live ECG stream and shows
real-time predictions from the FastAPI backend.

Upgrades in this version:
  1. Timing between beats uses the ACTUAL RR interval (scaled), so faster
     heart rhythms stream faster and slower rhythms stream slower — just
     like a real patient.
  2. A live "Heart Rate (BPM)" readout, calculated from the RR interval.
  3. A live latency counter showing real inference time per beat.
  4. The actual ECG waveform snippet around the current beat is plotted,
     not just the abnormal-probability line — this looks like a real
     ECG monitor instead of just a chart of numbers.

Run with:
    streamlit run dashboard.py

Make sure the FastAPI server (api.py) is already running in another
terminal at http://127.0.0.1:8000 before starting this.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import time
import os
import wfdb

API_URL = "http://127.0.0.1:8000/predict"
FEATURES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "ecg_features.csv")
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mitbih_data")

MAX_ROWS_SHOWN = 10
MAX_ALERTS_SHOWN = 5
N_BEATS_TO_SIMULATE = 50
WAVEFORM_WINDOW = 90        # must match WINDOW in feature_extraction.py

# Real RR intervals are ~0.6-1.0s; to keep a demo from running too long,
# we compress real time by this factor (e.g. 4 = 4x faster than real life).
TIME_COMPRESSION = 4

st.set_page_config(page_title="Real-Time Heartbeat Monitor", layout="wide")
st.title("Real-Time Abnormal Heartbeat Detection Dashboard")

df = pd.read_csv(FEATURES_CSV)

st.write("Simulating a live stream of heartbeats using each beat's actual "
         "RR interval for timing, with live heart rate, latency, and ECG "
         "waveform display.")

start = st.button("Start Live Simulation")

# --- Fixed layout: status/BPM/latency row first, then alerts, then charts ---
metrics_placeholder = st.empty()
alert_log_placeholder = st.empty()
waveform_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

history = []
alert_log = []
raw_record_cache = {}  # cache loaded wfdb records so we don't reload from disk each beat


def get_raw_signal(record_name):
    """Load (and cache) the raw ECG signal for a record."""
    if record_name not in raw_record_cache:
        record_path = os.path.join(RAW_DATA_DIR, str(record_name))
        record = wfdb.rdrecord(record_path)
        raw_record_cache[record_name] = record.p_signal[:, 0]
    return raw_record_cache[record_name]


if start:
    metrics_placeholder.info("Simulation starting...")

    for i, row in df.head(N_BEATS_TO_SIMULATE).iterrows():
        payload = {
            "rr_prev": row["rr_prev"],
            "rr_next": row["rr_next"],
            "amplitude": row["amplitude"],
            "local_mean": row["local_mean"],
            "local_std": row["local_std"],
        }

        # --- Measure real round-trip latency for this beat ---
        t0 = time.perf_counter()
        try:
            response = requests.post(API_URL, json=payload, timeout=2)
            result = response.json()
        except Exception as e:
            metrics_placeholder.error(f"Could not reach API: {e}")
            break
        latency_ms = (time.perf_counter() - t0) * 1000

        # --- Live heart rate from the RR interval ---
        rr_interval_sec = row["rr_next"]
        bpm = 60.0 / rr_interval_sec if rr_interval_sec > 0 else 0

        history.append({
            "beat": i,
            "prediction": result["prediction"],
            "abnormal_probability": result["abnormal_probability"]
        })

        # --- Top row: current status + live vitals (BPM, latency) ---
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("Heart Rate", f"{bpm:.0f} BPM")
            col2.metric("Inference Latency", f"{latency_ms:.2f} ms")
            col3.metric("Beat #", f"{i}")

            if result["prediction"] == "Abnormal":
                st.error(
                    f"ALERT: Abnormal heartbeat detected at beat #{i} "
                    f"(probability: {result['abnormal_probability']})"
                )
                alert_log.append(
                    f"Beat #{i}: Abnormal (probability: {result['abnormal_probability']})"
                )
            else:
                st.success(
                    f"Beat #{i}: Normal (probability: {result['abnormal_probability']})"
                )

        # --- Recent alerts log ---
        if alert_log:
            recent_alerts = alert_log[-MAX_ALERTS_SHOWN:][::-1]
            alert_log_placeholder.warning(
                "Recent alerts:\n" + "\n".join(f"- {a}" for a in recent_alerts)
            )

        # --- Real ECG waveform snippet for this beat ---
        try:
            record_name = row["record"]
            sample_idx = int(row["sample_index"])
            signal = get_raw_signal(record_name)

            start_idx = max(0, sample_idx - WAVEFORM_WINDOW)
            end_idx = min(len(signal), sample_idx + WAVEFORM_WINDOW)
            waveform_segment = signal[start_idx:end_idx]

            waveform_df = pd.DataFrame({
                "sample": np.arange(len(waveform_segment)),
                "amplitude": waveform_segment
            })

            waveform_color = "#E24B4A" if result["prediction"] == "Abnormal" else "#5B9BD5"

            waveform_chart = (
                alt.Chart(waveform_df)
                .mark_line(color=waveform_color)
                .encode(
                    x=alt.X("sample:Q", title="Sample (within beat window)"),
                    y=alt.Y("amplitude:Q", title="ECG amplitude (mV)")
                )
                .properties(height=200, title=f"ECG waveform — beat #{i} ({result['prediction']})")
            )
            with waveform_placeholder.container():
                st.altair_chart(waveform_chart, use_container_width=True)
        except Exception:
            pass  # if raw record isn't available, skip the waveform silently

        # --- Abnormal probability trend chart with threshold line ---
        history_df = pd.DataFrame(history)

        with chart_placeholder.container():
            line = (
                alt.Chart(history_df)
                .mark_line(color="#5B9BD5")
                .encode(
                    x=alt.X("beat:Q", title="Beat number"),
                    y=alt.Y("abnormal_probability:Q", title="Abnormal probability",
                            scale=alt.Scale(domain=[0, 1])),
                )
            )
            points = (
                alt.Chart(history_df)
                .mark_circle(size=80)
                .encode(
                    x="beat:Q",
                    y="abnormal_probability:Q",
                    color=alt.Color(
                        "prediction:N",
                        scale=alt.Scale(domain=["Normal", "Abnormal"],
                                         range=["#5B9BD5", "#E24B4A"]),
                        legend=alt.Legend(title="Prediction")
                    ),
                    tooltip=["beat", "prediction", "abnormal_probability"]
                )
            )
            threshold_line = (
                alt.Chart(pd.DataFrame({"y": [0.5]}))
                .mark_rule(color="#888880", strokeDash=[4, 4])
                .encode(y="y:Q")
            )
            combined_chart = (
                (line + points + threshold_line)
                .properties(height=280, title="Live abnormal heartbeat probability")
            )
            st.altair_chart(combined_chart, use_container_width=True)

        with table_placeholder.container():
            st.dataframe(history_df.tail(MAX_ROWS_SHOWN), use_container_width=True)

        # --- Sleep for the ACTUAL RR interval (compressed for demo speed) ---
        time.sleep(max(0.05, rr_interval_sec / TIME_COMPRESSION))

    metrics_placeholder.info("Simulation complete.")