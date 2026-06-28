Task 8 — The End-to-End Pipeline
PlaceMux · Altrodav Technologies · Phase 1 Industry Immersion
Objective
Assemble every stage from Tasks 5–7 into a single, one-command, reproducible ML pipeline: data → features → preprocessing → model → evaluation → saved artifacts.
How to Run
bashpip install scikit-learn joblib matplotlib openpyxl pandas
python task8_pipeline.py
That's it. One command. All stages run automatically.
Pipeline Stages
StageWhat happens1. Data LoadingLoads 6 xlsx files, validates schema, fails loudly on missing columns2. Target EngineeringBinarizes difficulty_level at median (42) → Easy=0, Hard=13. Train/Val/Test Split70/15/15 stratified split, random_state=424. Feature Engineering11 features derived; aggregate features computed from train-only (leakage-safe)5. sklearn PipelineColumnTransformer (scaler + passthrough) chained with RandomForestClassifier6. EvaluationConfusion matrix, precision/recall/F1, ROC — threshold=0.29 (Task 6 cost reasoning)7. Save ArtifactsModel, metrics, experiment log saved to disk8. Reproducibility CheckReloads saved model, re-predicts, compares SHA256 hash
Pipeline Design
Why preprocessing lives inside the Pipeline:

Chaining ColumnTransformer + RandomForestClassifier into one sklearn.Pipeline means the scaler is fit once on training data and automatically applied the same way at inference. This eliminates the most common production bug — accidentally re-fitting the scaler on new data.
Why aggregate features are outside the Pipeline:

topic_avg_difficulty and domain_avg_difficulty require a groupby on the training labels. sklearn Pipelines don't support this natively without a custom transformer. Instead, we compute them from train-only data before the pipeline, freeze the lookup tables, and map them to val/test — the correct leakage-safe pattern.
Dataset

6 xlsx files, 4,799 rows, 6 domains
Target: difficulty_level binarized at 42 → 48.6% Easy / 51.4% Hard

Results
SplitAccuracyPrecision (Hard)Recall (Hard)F1 (Hard)Validation0.610.580.860.692Test (unseen)0.620.590.860.703
Threshold: 0.29 — tuned for high recall on Hard class (Task 6 cost reasoning).
Reproducibility
Every run produces identical predictions. Verified via SHA256 hash of test predictions — original model and reloaded .joblib produce byte-identical output.
Artifacts
FileDescriptiontask8_pipeline.pyFull one-command pipeline script with detailed commentstask8_model.joblibTrained sklearn Pipeline — load with joblib.load()task8_metrics.jsonEvaluation metrics for this runtask8_experiment_log.jsonFull run log with reproducibility hashtask8_pipeline.pngConfusion matrix + ROC curve + feature importance
Load the Saved Model
pythonimport joblib
pipeline = joblib.load("task8_model.joblib")
predictions = pipeline.predict(X_new)   # X_new must have the 11 baseline features
Stack

Python 3.12
scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier, metrics)
joblib (model serialisation)
pandas, matplotlib, openpyxl
