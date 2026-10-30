"""
train_model.py
===============
Trains a baseline classifier (COPD / SMOKERS / CONTROL) on patient_features.csv
using Leave-One-Patient-Out cross-validation (LOPO), since with only 34
patients a normal train/test split would waste data and give an unstable
estimate of accuracy.

Feature selection happens INSIDE each CV fold (fit only on the training
patients) to avoid leaking test-patient information into the feature choice --
a very easy mistake to make with this few samples and this many features.

Outputs:
  - printed LOPO accuracy + confusion matrix + per-class report
  - feature_importance.csv: which channels/stats separate the classes best
    (fit on ALL data, for interpretation only -- not used in the CV score)
  - trained_model.joblib: final model fit on all 34 patients, for later use
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

DATA_PATH = Path("patient_features.csv")
OUTPUT_DIR = Path(".")
N_FEATURES_TO_SELECT = 10   # keep this small given only 34 patients total

MIN_SAMPLES_PER_CLASS = 8   # below this, LOPO results aren't trustworthy


def check_class_counts(y):
    """Warn loudly (don't silently proceed) if any class has too few samples
    to produce a meaningful accuracy estimate. A model CAN still be fit below
    this threshold, but the reported accuracy will be close to meaningless --
    e.g. 1 sample in a class means that class is either 0% or 100% correct
    with no in-between, telling you nothing about real-world performance."""
    counts = pd.Series(y).value_counts()
    print("\nClass counts:", counts.to_dict())

    thin_classes = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(thin_classes) > 0:
        print("\n" + "=" * 70)
        print("WARNING: the following classes have fewer than "
              f"{MIN_SAMPLES_PER_CLASS} samples:")
        for cls, n in thin_classes.items():
            print(f"  - {cls}: {n} sample(s)")
        print("Any accuracy/confusion-matrix numbers below involving these "
              "classes are NOT reliable -- with this few samples, one lucky "
              "or unlucky prediction swings the reported score by a large "
              "margin, and the model is likely to be overfit to these few "
              "examples rather than learning a real pattern. Treat this run "
              "as a pipeline test, not a valid accuracy claim, until you "
              "collect more samples for these classes.")
        print("=" * 70)
    return counts


def load_data(data_path):
    df = pd.read_csv(data_path)
    y = df["class"].values
    # feature columns = everything that isn't an ID/demographic/notes/timestamp field
    non_feature_cols = ["patient_id", "class", "gender", "age", "severity",
                         "main_cause", "smoking_years", "notes", "timestamp"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    # Keep X as a DataFrame (not .to_numpy()) so the fitted pipeline records
    # feature_names_in_ -- this lets the Django view check "does the live
    # sensor payload actually have the columns this model needs" instead of
    # failing with a confusing AttributeError.
    X = df[feature_cols].astype(float)
    return X, y, feature_cols, df


def run_lopo_cv(X, y):
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=N_FEATURES_TO_SELECT)),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(pred[0])

    return np.array(y_true_all), np.array(y_pred_all)


def fit_final_model_and_importances(X, y, feature_cols):
    """Fit on ALL data for interpretation + deployment. Not used for the CV score above."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES_TO_SELECT)),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")),
    ])
    pipe.fit(X, y)

    selected_mask = pipe.named_steps["select"].get_support()
    selected_features = [f for f, keep in zip(feature_cols, selected_mask) if keep]
    importances = pipe.named_steps["clf"].feature_importances_

    imp_df = pd.DataFrame({
        "feature": selected_features,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return pipe, imp_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=str(DATA_PATH),
        help="Path to the training CSV (e.g. patient_features.csv or "
             "patient_features_live.csv). Defaults to the lab dataset."
    )
    args = parser.parse_args()

    X, y, feature_cols, df = load_data(args.data)
    print(f"Loaded {X.shape[0]} patients, {X.shape[1]} candidate features, "
          f"from {args.data}")

    counts = check_class_counts(y)
    if counts.min() < 2:
        print("\nERROR: at least one class has fewer than 2 samples. "
              "Leave-One-Out CV cannot even run meaningfully (a class with "
              "1 sample can never be predicted correctly under LOPO, since "
              "the single example is always the one held out). Collect at "
              "least 2 samples per class before running this script.")
        return

    print("\nRunning Leave-One-Patient-Out cross-validation...")
    y_true, y_pred = run_lopo_cv(X, y)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nLOPO accuracy: {acc:.3f}  ({(y_true == y_pred).sum()}/{len(y_true)} correct)")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    labels = sorted(set(y))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("\nFitting final model on all data + computing feature importance...")
    final_model, imp_df = fit_final_model_and_importances(X, y, feature_cols)
    print(imp_df.to_string(index=False))

    imp_path = OUTPUT_DIR / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"\nSaved feature importance -> {imp_path}")

    model_path = OUTPUT_DIR / "trained_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved trained model -> {model_path}")


if __name__ == "__main__":
    main()
