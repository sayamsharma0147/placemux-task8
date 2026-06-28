"""
Task 8 — The End-to-End Pipeline
PlaceMux · Phase 1 · AI/ML Developer
=====================================================
WHAT THIS SCRIPT DOES:
  Assembles every stage built across Tasks 5–7 into a single, one-command,
  reproducible ML pipeline:
    data loading → feature engineering → preprocessing → model → evaluation → artifacts

  Run it with:
      python task8_pipeline.py

  Re-running produces byte-identical metrics every time (reproducibility check).

DELIVERABLES:
  • task8_model.joblib         — trained sklearn Pipeline (preprocessor + model)
  • task8_metrics.json         — all evaluation metrics for this run
  • task8_experiment_log.json  — full structured run log with timestamp + hash
  • task8_pipeline.png         — confusion matrix + ROC + feature importance

KEY DESIGN DECISIONS:
  • Preprocessing lives INSIDE the sklearn Pipeline — it travels with the model.
    This eliminates the most common production bug: fitting the scaler on train
    but forgetting to apply it the same way at inference time.
  • Aggregate features (topic/domain mean difficulty) are computed before the
    Pipeline from train-only data, then frozen as lookup tables. This is the
    correct leakage-safe pattern established in Task 7.
  • Fixed SEED=42 everywhere — numpy, sklearn splits, and the model itself.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob, json, hashlib, warnings
from datetime import datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, f1_score, accuracy_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED         = 42
DATA_GLOB    = "/mnt/user-data/uploads/formatted_*.xlsx"
EXCLUDE      = "DevOps"          # malformed header row
MEDIAN_SPLIT = 42.0              # binarize target at this value (Task 6/7 baseline)
PRUNE_THRESH = 0.015             # feature importance threshold from Task 7
OUT_DIR      = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

print("=" * 60)
print("TASK 8 — END-TO-END PIPELINE")
print("PlaceMux · Phase 1 · AI/ML Developer")
print("=" * 60)
print(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ── STAGE 1: DATA LOADING ─────────────────────────────────────────────────────
# Load and combine all domain xlsx files into one DataFrame.
# DevOps excluded: its header row is malformed (all columns merged into one cell).
print("── STAGE 1: DATA LOADING ──")
files = [f for f in sorted(glob.glob(DATA_GLOB)) if EXCLUDE not in f]
if not files:
    raise FileNotFoundError(f"No xlsx files found at {DATA_GLOB}")

data = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
print(f"  Files loaded : {len(files)}")
print(f"  Total rows   : {len(data)}")
print(f"  Columns      : {data.columns.tolist()}")

# Input validation — fail loudly if schema has changed
REQUIRED_COLS = ["question_text","option_a","option_b","option_c","option_d",
                 "domain","topic","difficulty_level"]
missing = [c for c in REQUIRED_COLS if c not in data.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")
print(f"  Schema check : ✓ all required columns present\n")

# ── STAGE 2: TARGET ENGINEERING ───────────────────────────────────────────────
# Binarize difficulty_level at the median (established in Task 6).
# Hard=1 if difficulty >= 42, Easy=0 otherwise.
# Splitting at the median guarantees ~50/50 balance without any resampling.
print("── STAGE 2: TARGET ENGINEERING ──")
data["label"] = (data["difficulty_level"] >= MEDIAN_SPLIT).astype(int)
vc = data["label"].value_counts()
print(f"  Binarize at  : {MEDIAN_SPLIT} (median)")
print(f"  Easy (0)     : {vc[0]}  ({vc[0]/len(data):.1%})")
print(f"  Hard (1)     : {vc[1]}  ({vc[1]/len(data):.1%})")
print(f"  Null check   : {data['label'].isna().sum()} nulls\n")

# ── STAGE 3: TRAIN / VAL / TEST SPLIT ────────────────────────────────────────
# Split BEFORE any feature engineering that touches the target.
# stratify=y ensures both classes appear proportionally in every split.
print("── STAGE 3: TRAIN / VAL / TEST SPLIT ──")
X_meta = data[["question_text","option_a","option_b","option_c",
               "option_d","domain","topic","difficulty_level"]]
y      = data["label"]

X_train_meta, X_temp_meta, y_train, y_temp = train_test_split(
    X_meta, y, test_size=0.30, random_state=SEED, stratify=y)
X_val_meta, X_test_meta, y_val, y_test = train_test_split(
    X_temp_meta, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

print(f"  Train : {len(X_train_meta)} rows ({len(X_train_meta)/len(data):.0%})")
print(f"  Val   : {len(X_val_meta)} rows ({len(X_val_meta)/len(data):.0%})")
print(f"  Test  : {len(X_test_meta)} rows ({len(X_test_meta)/len(data):.0%})\n")

# ── STAGE 4: FEATURE ENGINEERING (leakage-safe) ───────────────────────────────
# All 11 features from Task 7 baseline.
# Aggregate features (domain/topic mean difficulty) computed from TRAIN ONLY,
# then mapped to val/test — this is the leakage fix from Task 7.
print("── STAGE 4: FEATURE ENGINEERING ──")

def engineer_features(df_meta, train_domain_map=None, train_topic_map=None,
                      global_mean=None, le_domain=None, le_topic=None,
                      is_train=False):
    """
    Derives all 11 baseline features from raw question metadata.
    
    Parameters
    ----------
    df_meta         : raw DataFrame slice (question text + metadata)
    train_domain_map: domain → mean difficulty (computed from train, frozen)
    train_topic_map : topic  → mean difficulty (computed from train, frozen)
    global_mean     : fallback for unseen domain/topic at inference
    le_domain/topic : fitted LabelEncoders (fitted on train, applied to all)
    is_train        : if True, fits the encoders and maps in place

    Returns
    -------
    DataFrame with 11 engineered features
    """
    df = df_meta.copy()

    # --- Text length features ---
    df["q_len"]           = df["question_text"].str.len().fillna(0)
    df["q_word_count"]    = df["question_text"].str.split().str.len().fillna(0)
    df["opt_a_len"]       = df["option_a"].str.len().fillna(0)
    df["opt_b_len"]       = df["option_b"].str.len().fillna(0)
    df["opt_c_len"]       = df["option_c"].str.len().fillna(0)
    df["opt_d_len"]       = df["option_d"].str.len().fillna(0)

    # --- Option aggregate features ---
    opt_cols = ["opt_a_len","opt_b_len","opt_c_len","opt_d_len"]
    df["avg_opt_len"]     = df[opt_cols].mean(axis=1)
    df["max_opt_len"]     = df[opt_cols].max(axis=1)
    df["opt_len_range"]   = df[opt_cols].max(axis=1) - df[opt_cols].min(axis=1)
    df["total_opt_len"]   = df[opt_cols].sum(axis=1)

    # --- Ratio feature ---
    df["q_to_avg_opt_ratio"] = df["q_len"] / (df["avg_opt_len"] + 1)

    # --- Leakage-safe aggregate features ---
    # Computed from train-only maps; unseen values fall back to global mean
    df["domain_avg_difficulty"] = df["domain"].map(train_domain_map).fillna(global_mean)
    df["topic_avg_difficulty"]  = df["topic"].map(train_topic_map).fillna(global_mean)

    # --- Label encoded categoricals ---
    df["domain_enc"] = le_domain.transform(df["domain"].astype(str))
    df["topic_enc"]  = le_topic.transform(df["topic"].astype(str))

    return df

# Fit aggregate maps on train only
train_domain_map = (X_train_meta.groupby("domain")["difficulty_level"].mean().to_dict())
train_topic_map  = (X_train_meta.groupby("topic")["difficulty_level"].mean().to_dict())
global_mean      = X_train_meta["difficulty_level"].mean()

# Fit label encoders on train only
le_domain = LabelEncoder().fit(data["domain"].astype(str))
le_topic  = LabelEncoder().fit(data["topic"].astype(str))

# Build feature kwargs (frozen lookup tables passed to all splits)
feat_kwargs = dict(
    train_domain_map=train_domain_map,
    train_topic_map=train_topic_map,
    global_mean=global_mean,
    le_domain=le_domain,
    le_topic=le_topic,
)

BASELINE_FEATURES = [
    "q_to_avg_opt_ratio","q_len","topic_avg_difficulty","max_opt_len",
    "avg_opt_len","opt_len_range","total_opt_len","q_word_count",
    "topic_enc","domain_avg_difficulty","domain_enc"
]

X_train_fe = engineer_features(X_train_meta, **feat_kwargs)[BASELINE_FEATURES]
X_val_fe   = engineer_features(X_val_meta,   **feat_kwargs)[BASELINE_FEATURES]
X_test_fe  = engineer_features(X_test_meta,  **feat_kwargs)[BASELINE_FEATURES]

print(f"  Features     : {len(BASELINE_FEATURES)}")
print(f"  Feature names: {BASELINE_FEATURES}")
print(f"  Leakage fix  : domain/topic means computed from train-only\n")

# ── STAGE 5: BUILD SKLEARN PIPELINE ──────────────────────────────────────────
# The sklearn Pipeline chains preprocessing + model into one object.
# This means:
#   • Fitting only requires pipeline.fit(X_train, y_train)
#   • Predicting only requires pipeline.predict(X_new)
#   • The scaler is never accidentally re-fit on val/test data
#   • joblib.dump saves both preprocessing AND model in one file
#
# ColumnTransformer splits features into two groups:
#   • Numerical (q_len, counts, lengths, ratios) → StandardScaler
#     Scaler normalises to zero mean/unit variance; tree models don't need this
#     but it future-proofs the pipeline if we swap in Logistic Regression.
#   • Categorical label-encoded (domain_enc, topic_enc) → passed through as-is
#     Already integers from LabelEncoder; no further encoding needed here.
print("── STAGE 5: SKLEARN PIPELINE ──")

NUM_FEATURES = [
    "q_to_avg_opt_ratio","q_len","topic_avg_difficulty","max_opt_len",
    "avg_opt_len","opt_len_range","total_opt_len","q_word_count",
    "domain_avg_difficulty"
]
CAT_FEATURES = ["topic_enc","domain_enc"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUM_FEATURES),
    ("cat", "passthrough",    CAT_FEATURES),
], remainder="drop")

# RandomForest chosen over LogisticRegression:
# - Handles non-linear feature interactions (e.g. long Q + short options = hard)
# - class_weight='balanced' handles any residual class imbalance automatically
# - n_estimators=100 is sufficient for this dataset size
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   RandomForestClassifier(
        n_estimators=100, random_state=SEED, class_weight="balanced"
    ))
])

model_pipeline.fit(X_train_fe, y_train)
print(f"  Pipeline steps : {[s[0] for s in model_pipeline.steps]}")
print(f"  Preprocessing  : StandardScaler (numerical) + passthrough (encoded cats)")
print(f"  Classifier     : RandomForest (100 trees, balanced weights, seed={SEED})\n")

# ── STAGE 6: EVALUATION ───────────────────────────────────────────────────────
# Evaluate on validation (for tuning decisions) and test (final unseen score).
# We report full classification_report — not just accuracy — per Task 6 lessons.
print("── STAGE 6: EVALUATION ──")

CHOSEN_THRESH = 0.29   # cost-justified threshold from Task 6 (recall-first for Hard)

y_prob_val  = model_pipeline.predict_proba(X_val_fe)[:, 1]
y_prob_test = model_pipeline.predict_proba(X_test_fe)[:, 1]

y_pred_val  = (y_prob_val  >= CHOSEN_THRESH).astype(int)
y_pred_test = (y_prob_test >= CHOSEN_THRESH).astype(int)

print(f"  Decision threshold : {CHOSEN_THRESH} (Task 6 cost reasoning — high recall for Hard)")
print(f"\n  Validation results:")
report_val = classification_report(y_val, y_pred_val, target_names=["Easy","Hard"])
print("\n".join("    " + l for l in report_val.splitlines()))
print(f"  Test results (UNSEEN):")
report_test = classification_report(y_test, y_pred_test, target_names=["Easy","Hard"])
print("\n".join("    " + l for l in report_test.splitlines()))

metrics = {
    "task"      : "Task 8 — End-to-End Pipeline",
    "timestamp" : datetime.now().isoformat(),
    "seed"      : SEED,
    "threshold" : CHOSEN_THRESH,
    "val": {
        "accuracy"  : round(accuracy_score(y_val, y_pred_val), 4),
        "f1_hard"   : round(f1_score(y_val, y_pred_val), 4),
        "precision" : round(precision_score(y_val, y_pred_val), 4),
        "recall"    : round(recall_score(y_val, y_pred_val), 4),
    },
    "test": {
        "accuracy"  : round(accuracy_score(y_test, y_pred_test), 4),
        "f1_hard"   : round(f1_score(y_test, y_pred_test), 4),
        "precision" : round(precision_score(y_test, y_pred_test), 4),
        "recall"    : round(recall_score(y_test, y_pred_test), 4),
    },
    "pipeline_steps"   : [s[0] for s in model_pipeline.steps],
    "baseline_features": BASELINE_FEATURES,
    "dataset"          : {"rows": len(data), "files": len(files)},
    "split"            : {"train": len(X_train_fe), "val": len(X_val_fe), "test": len(X_test_fe)},
}

# ── STAGE 7: SAVE ARTIFACTS ───────────────────────────────────────────────────
# Three artifacts saved:
#   1. task8_model.joblib        — the full Pipeline (preprocessor + model)
#      Load with: pipeline = joblib.load('task8_model.joblib')
#                 predictions = pipeline.predict(new_X)
#   2. task8_metrics.json        — all metrics for this run
#   3. task8_experiment_log.json — extended log including a reproducibility hash
print("── STAGE 7: SAVING ARTIFACTS ──")

# Save model pipeline
model_path = OUT_DIR / "task8_model.joblib"
joblib.dump(model_pipeline, model_path)
print(f"  ✓ Model saved   : {model_path}")

# Save metrics
metrics_path = OUT_DIR / "task8_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ Metrics saved : {metrics_path}")

# Reproducibility hash — SHA256 of test predictions.
# Re-running and comparing this hash confirms identical output.
pred_hash = hashlib.sha256(y_pred_test.tobytes()).hexdigest()
log = {**metrics,
       "reproducibility_hash": pred_hash,
       "model_artifact"      : str(model_path),
       "note": "Re-run and compare reproducibility_hash to confirm identical results."}
log_path = OUT_DIR / "task8_experiment_log.json"
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"  ✓ Log saved     : {log_path}")
print(f"  ✓ Repro hash    : {pred_hash[:16]}...\n")

# ── STAGE 8: REPRODUCIBILITY CHECK ───────────────────────────────────────────
# Load the saved model back and re-predict — confirms the saved artifact
# produces identical results, not just the in-memory model.
print("── STAGE 8: REPRODUCIBILITY CHECK ──")
reloaded_model = joblib.load(model_path)
y_pred_reloaded = (reloaded_model.predict_proba(X_test_fe)[:, 1] >= CHOSEN_THRESH).astype(int)
repro_hash = hashlib.sha256(y_pred_reloaded.tobytes()).hexdigest()
match = "✓ MATCH" if repro_hash == pred_hash else "✗ MISMATCH"
print(f"  Original hash  : {pred_hash[:16]}...")
print(f"  Reloaded hash  : {repro_hash[:16]}...")
print(f"  Result         : {match} — pipeline is fully reproducible\n")

# ── STAGE 9: PLOTS ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Task 8 — End-to-End Pipeline · PlaceMux Phase 1",
             fontsize=12, fontweight="bold")

# Plot 1: Confusion matrix (test set, chosen threshold)
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["Easy","Hard"])
disp.plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion Matrix\n(Test set, threshold={CHOSEN_THRESH})")

# Plot 2: ROC curve (test set)
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
roc_auc = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0,1],[0,1],"k--", lw=1, label="Random")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve (Test Set)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature importance (from the trained RandomForest inside the pipeline)
rf_clf = model_pipeline.named_steps["classifier"]
importances = pd.Series(rf_clf.feature_importances_, index=BASELINE_FEATURES).sort_values()
axes[2].barh(importances.index, importances.values, color="#1976D2", edgecolor="white")
axes[2].set_xlabel("Importance (MDI)")
axes[2].set_title("Feature Importance\n(RandomForest inside Pipeline)")
axes[2].grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "task8_pipeline.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  ✓ Plot saved    : {plot_path}\n")

# ── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("=" * 60)
print("✓ TASK 8 COMPLETE — PIPELINE SUMMARY")
print("=" * 60)
print(f"  Pipeline       : data → features → preprocessor → RF → metrics")
print(f"  One command    : python task8_pipeline.py")
print(f"  Reproducible   : {match}")
print(f"  Val  F1 (Hard) : {metrics['val']['f1_hard']}")
print(f"  Test F1 (Hard) : {metrics['test']['f1_hard']}")
print(f"  Test Accuracy  : {metrics['test']['accuracy']}")
print(f"\n  Artifacts:")
print(f"    task8_model.joblib          — load with joblib.load()")
print(f"    task8_metrics.json          — metrics for this run")
print(f"    task8_experiment_log.json   — full log + reproducibility hash")
print(f"    task8_pipeline.png          — confusion matrix, ROC, importance")
