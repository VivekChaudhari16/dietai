"""
train_model.py
----------------
Loads data/raw_data.csv, performs feature engineering, trains and compares
RandomForest, XGBoost, and GradientBoosting classifiers using cross-validation
and hyperparameter tuning (GridSearchCV), then saves the best model + all
preprocessing artifacts to the models/ directory for the Streamlit app.

Run:
    python src/train_model.py
"""

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/raw_data.csv"
MODELS_DIR = "models"


def load_and_engineer_features(path=DATA_PATH):
    """Load the raw CSV and build the exact feature set the model trains on.

    IMPORTANT: app.py recreates this same logic (bins, formulas, encoders) at
    prediction time. If you change anything here, update app.py to match.
    """
    df = pd.read_csv(path)

    # Median imputation for any missing numeric readings (mirrors how a real
    # Kaggle health dataset would need cleaning — median is robust to outliers
    # compared to mean).
    for col in ["Cholesterol", "BloodSugar", "SystolicBP"]:
        df[col] = df[col].fillna(df[col].median())

    # ---- Feature engineering ----
    # Convert continuous values into clinically-meaningful risk bands. Trees
    # (RF/XGBoost/GBM) can learn thresholds on their own, but handing them
    # pre-binned categories as an EXTRA feature often boosts accuracy since
    # it encodes domain knowledge directly rather than making the model
    # rediscover the clinical cutoffs from data alone.

    # BMI category — standard WHO clinical bins
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # Cholesterol risk band — standard clinical cutoffs (mg/dL)
    df["Cholesterol_Risk"] = pd.cut(
        df["Cholesterol"],
        bins=[0, 200, 240, 500],
        labels=["Normal", "Borderline", "High"],
    )

    # Blood sugar risk band — ADA (American Diabetes Association) fasting
    # glucose cutoffs: <100 normal, 100-125 prediabetic, >=126 diabetic
    df["Sugar_Risk"] = pd.cut(
        df["BloodSugar"],
        bins=[0, 100, 126, 500],
        labels=["Normal", "Prediabetic", "Diabetic"],
    )

    # Interaction feature: a single composite "metabolic load" score.
    # Each metric is divided by its own dataset mean (normalizing away the
    # different units/scales of BMI vs mg/dL) then summed — so someone who's
    # simultaneously high on all three ends up with a much higher score than
    # someone who's only elevated on one.
    df["Metabolic_Load"] = (
        (df["BMI"] / df["BMI"].mean())
        + (df["Cholesterol"] / df["Cholesterol"].mean())
        + (df["BloodSugar"] / df["BloodSugar"].mean())
    )

    # Encode all categorical/text columns to integers — required because
    # sklearn/XGBoost models only accept numeric input. We keep every fitted
    # LabelEncoder in a dict so app.py can reuse the EXACT same mapping at
    # prediction time (e.g. "Male" -> 1 must always mean the same thing).
    encoders = {}
    for col in ["Gender", "ActivityLevel", "BMI_Category", "Cholesterol_Risk", "Sugar_Risk"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Final list + ORDER of columns fed into the model. This exact list is
    # saved to models/feature_cols.pkl and re-used by app.py to build
    # prediction rows in the right column order.
    feature_cols = [
        "Age", "Gender_enc", "Height_cm", "Weight_kg", "BMI",
        "Cholesterol", "BloodSugar", "SystolicBP", "ActivityLevel_enc",
        "BMI_Category_enc", "Cholesterol_Risk_enc", "Sugar_Risk_enc",
        "Metabolic_Load",
    ]

    # Encode the target label (the 7 diet plan names) into integers 0-6 for
    # classification. Stored as "target" encoder so predictions can be
    # decoded back to a human-readable diet name later.
    target_le = LabelEncoder()
    df["target"] = target_le.fit_transform(df["DietRecommendation"])
    encoders["target"] = target_le

    X = df[feature_cols]  # model inputs
    y = df["target"]      # model output (diet class, encoded as int)

    return X, y, encoders, feature_cols, df


def train_and_compare(X, y):
    """Train & tune 3 candidate models, then return the best one by test F1."""

    # 80/20 train/test split. stratify=y keeps the same class proportions in
    # both splits — important here since some diet classes are rarer than
    # others; without stratification a random split could under-represent them.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features to zero mean / unit variance. Tree models (RF/XGBoost/
    # GBM) don't strictly need this, but it's included so the same
    # preprocessing pipeline would also work if you swap in a
    # scale-sensitive model (e.g. logistic regression, SVM, KNN) later.
    # fit_transform on TRAIN only, then transform (not fit) on TEST — fitting
    # the scaler on test data would leak test-set statistics into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5-fold stratified cross-validation used inside GridSearchCV below —
    # each hyperparameter combo is trained/validated 5 times on different
    # folds of the TRAINING set, giving a more reliable score than a single
    # train/validation split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # NOTE: these grids are intentionally moderate so the full pipeline trains
    # in a few minutes on a laptop. Widen them (more values per param) if you
    # want a more exhaustive search and have time/compute to spare.
    param_grids = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [150, 250],
                "max_depth": [8, 14, None],
                "min_samples_split": [2, 5],
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, eval_metric="mlogloss", use_label_encoder=False, n_jobs=-1
            ),
            "params": {
                "n_estimators": [150, 250],
                "max_depth": [4, 6],
                "learning_rate": [0.05, 0.1],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 150],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        },
    }

    results = {}
    fitted_models = {}

    # Loop over each candidate model, run GridSearchCV to find its best
    # hyperparameters via cross-validation, then evaluate that best version
    # on the held-out test set (data it has never seen in any form).
    for name, spec in param_grids.items():
        print(f"\n=== Tuning {name} ===")
        t0 = time.time()
        grid = GridSearchCV(
            spec["model"], spec["params"], cv=cv, scoring="f1_weighted",
            n_jobs=-1, verbose=0,
        )
        grid.fit(X_train_scaled, y_train)
        elapsed = time.time() - t0

        # grid.best_estimator_ is the model already re-fit on ALL training
        # data using the best hyperparameter combination found.
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        # weighted F1 accounts for class imbalance (some diets are rarer)
        # better than plain accuracy would.
        f1 = f1_score(y_test, preds, average="weighted")

        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} CV f1_weighted: {grid.best_score_:.4f} | Test acc: {acc:.4f} | Test f1: {f1:.4f} | time: {elapsed:.1f}s")

        results[name] = {
            "best_params": grid.best_params_,
            "cv_f1_weighted": grid.best_score_,
            "test_accuracy": acc,
            "test_f1_weighted": f1,
            "train_time_sec": elapsed,
        }
        fitted_models[name] = best_model

    # Pick the overall winner by test-set weighted F1 (not just accuracy,
    # since F1 better reflects performance across imbalanced classes).
    best_name = max(results, key=lambda k: results[k]["test_f1_weighted"])
    best_model = fitted_models[best_name]
    print(f"\n>>> BEST MODEL: {best_name} <<<")
    print(classification_report(
        y_test, best_model.predict(X_test_scaled)
    ))

    return best_name, best_model, fitted_models, results, scaler, (X_test, y_test)


def main():
    """Full pipeline entry point: load data -> engineer features -> train/
    compare 3 models -> persist the winner + preprocessing artifacts to disk
    so app.py can load them at runtime without retraining."""
    print("Loading data & engineering features...")
    X, y, encoders, feature_cols, df = load_and_engineer_features()
    print(f"Dataset shape: {X.shape}, classes: {y.nunique()}")

    best_name, best_model, fitted_models, results, scaler, test_data = train_and_compare(X, y)

    # Persist everything the Streamlit app needs at inference time:
    #   - the winning model itself
    #   - the fitted scaler (must reuse, never refit on new data)
    #   - all LabelEncoders (categorical -> int mappings, incl. target labels)
    #   - the exact feature column order the model expects
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(encoders, f"{MODELS_DIR}/encoders.pkl")
    joblib.dump(feature_cols, f"{MODELS_DIR}/feature_cols.pkl")

    # Save cross-val/test metrics for all 3 models so the "Model Performance"
    # page in app.py can display a full comparison, not just the winner.
    with open(f"{MODELS_DIR}/model_results.json", "w") as f:
        json.dump({
            "best_model": best_name,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved best model ({best_name}) and artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
