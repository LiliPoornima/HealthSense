# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os



# Path relative to preprocessing.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "health_lifestyle_classification.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "preprocessed_dataset.csv")


# Load dataset
df = pd.read_csv(RAW_PATH)

# Drop unnecessary columns
df = df.drop(columns=["electrolyte_level", "gene_marker_flag"], errors='ignore')

# Split columns by dtype
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Handle missing categorical values
high_missing_cats = ["alcohol_consumption", "exercise_type", "caffeine_intake"]
for col in high_missing_cats:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Numeric imputer (median)
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical imputer (mode/most frequent)
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Handle outliers
for col in ["age","height","weight","bmi","blood_pressure","heart_rate"]:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])

# Encode target
df["target"] = df["target"].map({"healthy":0, "diseased":1})

# Fix categorical types
for col in cat_cols:
    if col != "target" and col in df.columns:
        df[col] = df[col].astype("category")

# Fix negative & round numeric columns
if "daily_supplement_dosage" in df.columns:
    df["daily_supplement_dosage"] = np.where(
        df["daily_supplement_dosage"] < 0,
        0,
        df["daily_supplement_dosage"].round(2)
    )

# Function to clean dataset (round and fix types)
def clean_dataset(df):
    int_cols = [
        "survey_code", "age", "height", "waist_size",
        "blood_pressure", "heart_rate", "sleep_hours",
        "work_hours", "daily_steps", "meals_per_day","physical_activity"
    ]
    float_cols = [
        "weight", "bmi", "bmi_estimated", "bmi_scaled", "bmi_corrected",
        "cholesterol", "glucose", "insulin", "calorie_intake",
        "sugar_intake", "alcohol_consumption", "water_intake", "screen_time",
        "stress_level", "mental_health_score", "environmental_risk_score",
        "daily_supplement_dosage", "caffeine_intake","income"
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    return df

df = clean_dataset(df)

# Save preprocessed dataset
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
print(f"Preprocessed dataset saved at {PROCESSED_PATH}")

# Optional: check class balance and missing values
print("Class balance:\n", df["target"].value_counts(normalize=True))
print("\nMissing values:\n", df.isnull().sum())
