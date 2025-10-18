# src/train_model.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from imblearn.over_sampling import SMOTE

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "preprocessed_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed CSV
df = pd.read_csv(PROCESSED_DATA_PATH)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

#Normalization

# Scale numeric features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
# 50% - 50%  
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Resampled target distribution:\n", pd.Series(y_train_res).value_counts(normalize=True))

# Define models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

# Train, evaluate, track best model
best_model_name = None
best_f1 = 0
best_model = None

for name, model in models.items():
    print("\n==============================")
    print(f"Training: {name}")
    print("==============================")
    
    model.fit(X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f1_class1 = f1_score(y_test, y_pred, pos_label=1)
    roc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (overall): {f1:.4f}")
    print(f"F1 Score (diseased=1): {f1_class1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    
    if f1_class1 > best_f1:
        best_f1 = f1_class1
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} (F1 for diseased = {best_f1:.4f})")

# Save best model
joblib_file = os.path.join(MODELS_DIR, f"best_model_{best_model_name}.pkl")
joblib.dump(best_model, joblib_file)
print(f"Best model saved as {joblib_file}")
