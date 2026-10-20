# Real-Time Abnormal Heartbeat Detection Using ECG Signals

A classical machine learning project that detects abnormal heartbeats
from ECG signals (MIT-BIH Arrhythmia Database) and streams live
predictions through a real-time dashboard.

## Project Structure

```
ecg-arrhythmia-project/
├── data/
│   ├── mitbih_data/         # downloaded raw ECG records (created automatically)
│   └── ecg_features.csv     # extracted features (created by feature_extraction.py)
├── models/
│   ├── scaler.pkl           # saved feature scaler
│   ├── best_model.pkl       # saved best-performing trained model
│   └── model_comparison.csv # metrics table for all models (use in your paper)
├── src/
│   ├── load_data.py         # Part 1: download ECG records
│   ├── feature_extraction.py# Part 2: extract ML features from beats
│   ├── train_models.py      # Part 3: train & compare models
│   ├── api.py                # Part 4: FastAPI real-time scoring service
│   └── dashboard.py          # Part 5: Streamlit live monitoring dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run (in order)

### Step 1 — Download data
```bash
cd src
python load_data.py
```
This downloads a few sample records by default (`TEST_RECORDS`). Once
this works, change the argument in `load_data.py` from `TEST_RECORDS` to
`ALL_RECORDS` if you need to download the full 48-record dataset.

### Step 2 — Extract features
```bash
python feature_extraction.py
```
This creates `data/ecg_features.csv` — one row per heartbeat from every
complete record found in `data/mitbih_data`, with engineered features
(RR intervals, amplitude, local signal stats) and a normal/abnormal label.

### Step 3 — Train and compare models
```bash
python train_models.py
```
Trains Logistic Regression, Random Forest, SVM, and XGBoost, prints
metrics for each, and saves the best model to `models/best_model.pkl`.
Also saves `models/model_comparison.csv` — use this table directly in
your paper's Results section.

### Step 4 — Start the real-time API (keep this terminal running)
```bash
uvicorn api:app --reload --port 8000
```

### Step 5 — Start the live dashboard (in a SECOND terminal, keep running)
```bash
streamlit run dashboard.py
```
Open the local URL Streamlit gives you, click "Start Live Simulation",
and watch predictions stream in beat-by-beat with live alerts.

## Notes

- This project uses classical machine learning only (no deep learning),
  which keeps the models fast, interpretable, and lightweight enough
  for real-time use.
- Beats labeled 'N', 'L', 'R', 'e', 'j' are treated as Normal; all
  other annotation symbols are treated as Abnormal. You can refine
  this grouping (e.g., multi-class instead of binary) as a next step.
- For your conference paper, compare `models/model_comparison.csv`
  against published MIT-BIH benchmark results to fill out your
  Results and Discussion sections.
