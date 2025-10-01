# src/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Base directory for relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# Load trained artifacts
model = joblib.load(os.path.join(MODELS_DIR, "best_model_Decision_Tree.pkl"))  
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

st.title("Dynamic Health Risk Prediction")
st.write("Predict whether a person is healthy or diseased based on their features.")

# Load dataset to get feature info
df = pd.read_csv(os.path.join(DATA_DIR, "preprocessed_dataset.csv"))
df_features = df.drop(columns=["target"], errors="ignore")

# Generate dynamic inputs
user_input = {}
for col in df_features.columns:
    dtype = df_features[col].dtype
    
    if np.issubdtype(dtype, np.integer):
        min_val = int(df_features[col].min())
        max_val = int(df_features[col].max() * 1.5)
        default_val = int(df_features[col].median())
        user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val, step=1)
    elif np.issubdtype(dtype, np.floating):
        min_val = float(df_features[col].min())
        max_val = float(df_features[col].max() * 1.5)
        default_val = float(df_features[col].median())
        user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val, step=0.01, format="%.2f")
    else:
        options = df_features[col].dropna().unique().tolist()
        if len(options) <= 2:
            user_input[col] = st.radio(col, options)
        else:
            user_input[col] = st.selectbox(col, options)

# Prepare input for prediction
input_df = pd.DataFrame([user_input])

# Scale numeric columns ONLY
num_cols = [col for col in df_features.select_dtypes(include=[np.number]).columns if col in input_df.columns]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# One-hot encode categorical columns (only those seen during training)
for col in df_features.select_dtypes(include=["object", "category"]).columns:
    for feature in feature_names:
        if feature.startswith(col + "_"):
            category = feature.replace(col + "_", "")
            input_df[feature] = (input_df[col] == category).astype(int)
    input_df.drop(columns=[col], inplace=True)

# Reindex to match training features exactly
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.write(f"Prediction: {'Diseased' if prediction==1 else 'Healthy'}")
    
    if prediction_proba is not None:
        risk_percent = prediction_proba * 100
        st.write(f"Risk Percentage: {risk_percent:.2f}%")

        # Determine risk category
        if risk_percent <= 30:
            risk_category = "Low risk: Likely Healthy"
        elif 30 < risk_percent <= 70:
            risk_category = "Moderate risk: Possibly Diseased"
        else:
            risk_category = "High risk: Likely Diseased"
        
        st.write(f"Risk Category: {risk_category}")

        # Recommendations
        recommendations = []

        # Numeric checks
        bmi = user_input.get("bmi")
        if isinstance(bmi, (int, float)) and bmi > 25:
            recommendations.append("Maintain a healthy BMI: Consider balanced diet and exercise.")

        bp = user_input.get("blood_pressure")
        if isinstance(bp, (int, float)) and bp > 120:
            recommendations.append("Monitor blood pressure: Reduce salt intake and manage stress.")

        # Categorical checks
        exercise = user_input.get("physical_activity")
        if isinstance(exercise, str) and exercise.lower() in ["low", "sedentary"]:
            recommendations.append("Increase physical activity: At least 30 mins/day of moderate exercise.")

        smoking = user_input.get("smoking_status")
        if isinstance(smoking, str) and smoking.lower() != "non-smoker":
            recommendations.append("Quit smoking to reduce health risks.")

        alcohol = user_input.get("alcohol_consumption")
        if isinstance(alcohol, str) and alcohol.lower() not in ["none", "never"]:
            recommendations.append("Limit alcohol consumption to reduce health risks.")

        caffeine = user_input.get("caffeine_intake")
        if isinstance(caffeine, str) and caffeine.lower() not in ["none", "low"]:
            recommendations.append("Moderate caffeine intake.")

        sunlight = user_input.get("sunlight_exposure")
        if isinstance(sunlight, str) and sunlight.lower() in ["low", "minimal"]:
            recommendations.append("Increase sunlight exposure for Vitamin D.")

        # Display recommendations
        if recommendations:
            st.write("💡 **Recommendations to reduce disease risk:**")
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("You are following healthy habits! Keep it up.")
