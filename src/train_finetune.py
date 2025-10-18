# src/train_finetune.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
import xgboost as xgb


#  Paths Setup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "preprocessed_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Load Preprocessed Data

df = pd.read_csv(DATA_PATH)
print(" Data loaded successfully:", df.shape)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()


# Feature Scaling & Encoding

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, "feature_names.pkl"))


# Split into Train/Test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Handle Class Imbalance

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(" Class balance after SMOTE:")
print(pd.Series(y_train_res).value_counts(normalize=True))


# Sampling for Faster Tuning

X_sample = X_train_res.sample(frac=0.2, random_state=42)
y_sample = y_train_res.loc[X_sample.index]
print("Using sample for tuning:", X_sample.shape)


# Model Fine-Tuning


# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
param_dist_logreg = {
    "C": uniform(0.001, 10),
    "penalty": ["l2"],
    "solver": ["lbfgs", "liblinear"]
}
print("\n🔧 Fine-tuning Logistic Regression...")
random_search_logreg = RandomizedSearchCV(
    log_reg,
    param_distributions=param_dist_logreg,
    n_iter=20,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=2
)
random_search_logreg.fit(X_sample, y_sample)
print("Best Logistic Regression params:", random_search_logreg.best_params_)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
param_dist_dt = {
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "criterion": ["gini", "entropy"]
}
print("\n Fine-tuning Decision Tree...")
random_search_dt = RandomizedSearchCV(
    dt,
    param_distributions=param_dist_dt,
    n_iter=20,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=2
)
random_search_dt.fit(X_sample, y_sample)
print(" Best Decision Tree params:", random_search_dt.best_params_)

# Random Forest
rf = RandomForestClassifier(random_state=42)
param_dist_rf = {
    "n_estimators": randint(100, 500),
    "max_depth": randint(5, 20),
    "min_samples_split": randint(2, 15),
    "min_samples_leaf": randint(1, 8),
    "bootstrap": [True, False]
}
print("\n Fine-tuning Random Forest...")
random_search_rf = RandomizedSearchCV(
    rf,
    param_distributions=param_dist_rf,
    n_iter=20,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=2
)
random_search_rf.fit(X_sample, y_sample)
print(" Best Random Forest params:", random_search_rf.best_params_)

# XGBoost
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
param_dist_xgb = {
    "n_estimators": randint(200, 500),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.7, 0.3),
    "colsample_bytree": uniform(0.7, 0.3)
}
print("\n Fine-tuning XGBoost...")
random_search_xgb = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist_xgb,
    n_iter=20,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42,
    verbose=2
)
random_search_xgb.fit(X_sample, y_sample)
print(" Best XGBoost params:", random_search_xgb.best_params_)


# Evaluate All Models

models = {
    "Logistic Regression": random_search_logreg.best_estimator_,
    "Decision Tree": random_search_dt.best_estimator_,
    "Random Forest": random_search_rf.best_estimator_,
    "XGBoost": random_search_xgb.best_estimator_
}

print("\n Evaluating all models on full test set...\n")
best_model_name = None
best_f1 = 0

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("-" * 50)

    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name


#  Save Best Model

best_model = models[best_model_name]
model_path = os.path.join(MODELS_DIR, f"best_model_{best_model_name.replace(' ', '_')}.pkl")
joblib.dump(best_model, model_path)

print(f"\n Best Model: {best_model_name} (F1 = {best_f1:.4f})")
print(f" Model saved successfully at: {model_path}")