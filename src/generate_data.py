"""
generate_data.py
-----------------
Generates a large, realistic synthetic health & diet dataset (10,000 rows)
that mirrors the schema of common Kaggle "diet recommendation" / "health"
datasets: BMI, cholesterol, blood sugar, activity level, age, gender, etc.

If you have a real Kaggle dataset, just drop your CSV into `data/raw_data.csv`
with matching column names and skip running this script -- train_model.py
will use whichever CSV is present.

To swap in a real Kaggle dataset instead:
    1. Go to kaggle.com and search "diet recommendation dataset" or
       "health BMI cholesterol dataset"
    2. Download the CSV
    3. Save it as data/raw_data.csv with columns matching this schema
       (or update COLUMN_MAP in train_model.py)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000  # large dataset size


def generate_dataset(n=N):
    """Build one synthetic health record per row, with realistic correlations
    between age/gender/height/weight/BMI/cholesterol/blood sugar so the
    resulting dataset behaves like real-world health data (not pure noise)."""
    age = np.random.randint(18, 75, n)
    gender = np.random.choice(["Male", "Female"], n)

    # Height in cm, weight in kg — drawn from a normal distribution centered
    # differently per gender (rough population averages), then clipped to a
    # realistic human range to avoid absurd outliers.
    height = np.where(
        gender == "Male",
        np.random.normal(172, 7, n),
        np.random.normal(160, 6, n),
    ).clip(140, 210)

    weight = np.where(
        gender == "Male",
        np.random.normal(78, 14, n),
        np.random.normal(65, 13, n),
    ).clip(35, 160)

    bmi = weight / ((height / 100) ** 2)  # standard BMI formula

    # Cholesterol trends upward with age and BMI in real populations, plus
    # random individual variation (np.random.normal noise term).
    cholesterol = (
        140
        + (age * 0.6)
        + (bmi * 1.8)
        + np.random.normal(0, 20, n)
    ).clip(120, 320)

    # blood sugar (fasting, mg/dL) correlates with bmi + age
    blood_sugar = (
        68
        + (bmi * 0.75)
        + (age * 0.25)
        + np.random.normal(0, 10, n)
    ).clip(65, 250)

    # Categorical activity level, sampled with weighted probabilities so
    # "Sedentary" and "Moderate" are most common (roughly matches real
    # population activity surveys).
    activity_level = np.random.choice(
        ["Sedentary", "Low", "Moderate", "High", "Very High"],
        n,
        p=[0.25, 0.2, 0.25, 0.2, 0.1],
    )

    # Numeric encoding of activity used only internally to drive the
    # blood-pressure formula and the diet-labeling rules below — NOT saved
    # to the final CSV (dropped at the end of this function).
    activity_score_map = {
        "Sedentary": 0, "Low": 1, "Moderate": 2, "High": 3, "Very High": 4
    }
    activity_score = np.array([activity_score_map[a] for a in activity_level])

    # Blood pressure rises with age/BMI but is slightly lowered by higher
    # activity levels — mirrors the real protective effect of exercise.
    systolic_bp = (
        110 + (age * 0.4) + (bmi * 0.5) - (activity_score * 2)
        + np.random.normal(0, 8, n)
    ).clip(90, 190)

    # ---- Diet label logic (mimics a clinical rule-of-thumb) ----
    # This is the "ground truth" the ML model later learns to approximate.
    # Rules are checked in priority order (most urgent health flag wins) —
    # e.g. a diabetic reading takes priority over a BMI-based recommendation.
    def assign_diet(row):
        b, c, s, act = row["BMI"], row["Cholesterol"], row["BloodSugar"], row["ActivityScore"]
        if s >= 126:
            return "Diabetic-Friendly Low-Carb Diet"
        if c >= 240:
            return "Heart-Healthy Low-Cholesterol Diet"
        if b >= 30:
            return "Weight-Loss Calorie-Deficit Diet"
        if b < 18.5:
            return "High-Calorie Weight-Gain Diet"
        if act <= 1 and b >= 25:
            return "Low-Impact Balanced Diet"
        if act >= 3:
            return "High-Protein Athletic Diet"
        return "Balanced Maintenance Diet"

    df = pd.DataFrame({
        "Age": age.astype(int),
        "Gender": gender,
        "Height_cm": height.round(1),
        "Weight_kg": weight.round(1),
        "BMI": bmi.round(2),
        "Cholesterol": cholesterol.round(1),
        "BloodSugar": blood_sugar.round(1),
        "SystolicBP": systolic_bp.round(1),
        "ActivityLevel": activity_level,
        "ActivityScore": activity_score,
    })

    df["DietRecommendation"] = df.apply(assign_diet, axis=1)

    # Inject a small amount of missing data (~1.5% per column) to mimic the
    # messiness of real-world/Kaggle datasets — train_model.py then handles
    # this via median imputation, which is worth demonstrating in the pipeline.
    for col in ["Cholesterol", "BloodSugar", "SystolicBP"]:
        mask = np.random.rand(n) < 0.015
        df.loc[mask, col] = np.nan

    df = df.drop(columns=["ActivityScore"])  # was only a helper for label logic, not a real feature
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/raw_data.csv", index=False)
    print(f"Generated dataset with {len(df)} rows -> data/raw_data.csv")
    print(df["DietRecommendation"].value_counts())
