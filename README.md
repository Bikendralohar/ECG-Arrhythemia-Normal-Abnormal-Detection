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
│   ├── best_model.pkl       # best inter-patient model, saved as a full sklearn Pipeline (scaler inside)
│   ├── model_meta.json      # model name, feature list, tuned threshold, DS1/DS2 records, CV + test metrics
│   ├── cv_summary_ds1.csv   # 5-fold patient-grouped CV on DS1 (mean +- std) - model selection
│   ├── model_comparison_both_splits.csv # intra-patient vs inter-patient (DS1 -> DS2)
│   ├── per_class_detection_ds2.csv      # N specificity, S/V/F sensitivity on DS2
│   ├── ds2_bootstrap_ci.csv             # 95% CIs (patient-level bootstrap)
│   ├── aami_multiclass_ds2.csv          # 4-class N/S/V/F: Se, +P, F1 per class
│   ├── ablation_ds2.csv                 # feature-set ablation
│   ├── permutation_importance.csv
│   ├── latency_benchmark.csv
│   └── figures/
├── src/
│   ├── config.py            # shared paths, feature list, label rules, split settings
│   ├── load_data.py         # Part 1: download ECG records
│   ├── feature_extraction.py# Part 2: extract ML features from beats
│   ├── train_models.py      # Part 3: CV model selection on DS1, test on DS2
│   ├── paper_experiments.py # CIs, AAMI 4-class, ablation, importance, ROC/PR figures
│   ├── qrs_detector.py      # causal streaming Pan-Tompkins QRS (R-peak) detector
│   ├── realtime_pipeline.py # raw ECG samples -> detected beats -> 6 features
│   ├── end_to_end_eval.py   # detector accuracy + classification on detected beats
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
This downloads any of the 48 records that are not already present.
Use `python load_data.py --test` to fetch only 3 records for a quick test.

### Step 2 — Extract features
```bash
python feature_extraction.py
```
This creates `data/ecg_features.csv` — one row per heartbeat from every
complete record found in `data/mitbih_data` (paced records 102, 104, 107,
217 excluded per AAMI EC57), with a normal/abnormal label. Only real beat
annotations are used; non-beat annotations (`+ ~ " | x ! [ ]`) are skipped.

Model features (all causal — none uses the next beat):
`rr_prev`, `rr_local_avg` (mean of last 10 RR), `rr_ratio`
(rr_prev / rr_local_avg), `amplitude`, `local_mean`, `local_std`.
`rr_next` is stored only for the dashboard's playback timing.

### Step 3 — Train and compare models
```bash
python train_models.py
```
Evaluation protocol (standard MIT-BIH inter-patient protocol):

1. **Model selection** — 5-fold patient-grouped cross-validation inside DS1.
   Each model's decision threshold is also tuned here (max F1 on out-of-fold
   predictions). DS2 is never used for any choice.
2. **Test** — every model is refit on all of DS1 and tested once on DS2
   (de Chazal et al., 2004: 22 training vs 22 different test patients).
3. **Contrast** — an intra-patient random split, to show how much a naive
   split overestimates performance.

The CV-selected model is saved as `models/best_model.pkl` + `models/model_meta.json`.
(Takes a few minutes — the SVM is the slow part.)

### Step 3b — Paper experiments, benchmarks and figures
```bash
python paper_experiments.py     # CIs, AAMI 4-class table, ablation, importance, ROC/PR
python generate_figures.py      # fig1 F1 comparison, fig2 confusion matrix
python latency_benchmark.py
python check_train_test_diff.py
```

### Step 3c — End-to-end evaluation with the QRS detector
```bash
python end_to_end_eval.py
```
Detects every beat in the raw ECG with `qrs_detector.py` (no reference beat
positions), reports detector sensitivity/positive predictivity (150 ms
tolerance, AAMI EC57), and evaluates the classifier on the detected beats.
Saves `models/qrs_detection_results.csv` and `models/end_to_end_results.csv`.

### Step 4 — Start the real-time API (keep this terminal running)
```bash
uvicorn api:app --reload --port 8000
```

### Step 5 — Start the live dashboard (in a SECOND terminal, keep running)
```bash
streamlit run dashboard.py
```
The dashboard feeds the raw ECG of a DS2 test patient in 100 ms chunks,
finds each beat with the streaming QRS detector and sends its features to the
API — reference annotations are only used to display the true label.
Open the local URL Streamlit gives you, pick one of the DS2 test
patients, click "Start Live Simulation", and watch predictions stream in
beat-by-beat with live alerts and the true label for comparison.

## Notes

- This project uses classical machine learning only (no deep learning),
  which keeps the models fast, interpretable, and lightweight enough
  for real-time use.
- Beats labeled 'N', 'L', 'R', 'e', 'j' are treated as Normal; all
  other beat symbols are treated as Abnormal. You can refine
  this grouping (e.g., multi-class instead of binary) as a next step.
- For your conference paper, compare `models/model_comparison_both_splits.csv`
  against published MIT-BIH benchmark results to fill out your
  Results and Discussion sections.
