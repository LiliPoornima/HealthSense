import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ===============================
# Setup paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "processed")

# Load model artifacts
model = joblib.load(os.path.join(MODELS_DIR, "best_model_Decision_Tree.pkl"))  
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Health Risk Predictor", page_icon="🩺", layout="wide")

st.title("🩺 Dynamic Health Risk Prediction")
st.write("Provide your health and lifestyle details to predict the likelihood of disease risk.")

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(os.path.join(DATA_DIR, "preprocessed_dataset.csv"))
df_features = df.drop(columns=["target"], errors="ignore")

# ===============================
# Sidebar Inputs
# ===============================
st.sidebar.header("📝 Enter Your Details")
user_input = {}

for col in df_features.columns:
    dtype = df_features[col].dtype
    with st.sidebar:
        if np.issubdtype(dtype, np.integer):
            user_input[col] = st.number_input(
                f"{col}", 
                min_value=int(df_features[col].min()), 
                max_value=int(df_features[col].max() * 1.5), 
                value=int(df_features[col].median()), 
                step=1
            )
        elif np.issubdtype(dtype, np.floating):
            user_input[col] = st.number_input(
                f"{col}", 
                min_value=float(df_features[col].min()), 
                max_value=float(df_features[col].max() * 1.5), 
                value=float(df_features[col].median()), 
                step=0.01, 
                format="%.2f"
            )
        else:
            options = df_features[col].dropna().unique().tolist()
            if len(options) <= 2:
                user_input[col] = st.radio(f"{col}", options)
            else:
                user_input[col] = st.selectbox(f"{col}", options)

# ===============================
# Data Preparation
# ===============================
input_df = pd.DataFrame([user_input])
num_cols = [col for col in df_features.select_dtypes(include=[np.number]).columns if col in input_df.columns]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# One-hot encoding
for col in df_features.select_dtypes(include=["object", "category"]).columns:
    for feature in feature_names:
        if feature.startswith(col + "_"):
            category = feature.replace(col + "_", "")
            input_df[feature] = (input_df[col] == category).astype(int)
    input_df.drop(columns=[col], inplace=True)

input_df = input_df.reindex(columns=feature_names, fill_value=0)

# ===============================
# Prediction
# ===============================
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns([1,2])

    with col1:
        st.metric("Prediction", "Diseased 🟥" if prediction==1 else "Healthy 🟩")

    with col2:
        if prediction_proba is not None:
            risk_percent = prediction_proba * 100
            st.progress(int(risk_percent))
            st.write(f"**Risk Score:** {risk_percent:.2f}%")

            if risk_percent <= 30:
                st.success("✅ Low risk: Likely Healthy")
            elif 30 < risk_percent <= 70:
                st.warning("⚠️ Moderate risk: Possibly Diseased")
            else:
                st.error("🚨 High risk: Likely Diseased")

    # ===============================
    # Recommendations
    # ===============================
    st.subheader("💡 Personalized Recommendations")
    recommendations = []

    # Numeric checks
    bmi = user_input.get("bmi")
    if isinstance(bmi, (int, float)) and bmi > 25:
        recommendations.append("⚖️ Maintain a healthy BMI: balanced diet + exercise.")

    bp = user_input.get("blood_pressure")
    if isinstance(bp, (int, float)) and bp > 120:
        recommendations.append("🫀 Monitor blood pressure: reduce salt & manage stress.")

    # Lifestyle checks
    exercise = user_input.get("physical_activity")
    if isinstance(exercise, str) and exercise.lower() in ["low", "sedentary"]:
        recommendations.append("🏃 Increase physical activity: 30 mins/day minimum.")

    smoking = user_input.get("smoking_status")
    if isinstance(smoking, str) and smoking.lower() != "non-smoker":
        recommendations.append("🚭 Quit smoking to reduce long-term risks.")

    alcohol = user_input.get("alcohol_consumption")
    if isinstance(alcohol, str) and alcohol.lower() not in ["none", "never"]:
        recommendations.append("🍷 Limit alcohol consumption.")

    caffeine = user_input.get("caffeine_intake")
    if isinstance(caffeine, str) and caffeine.lower() not in ["none", "low"]:
        recommendations.append("☕ Moderate caffeine intake.")

    sunlight = user_input.get("sunlight_exposure")
    if isinstance(sunlight, str) and sunlight.lower() in ["low", "minimal"]:
        recommendations.append("🌞 Increase sunlight exposure for Vitamin D.")

    if recommendations:
        for rec in recommendations:
            st.write(f"- {rec}")
    else:
        st.success("🎉 You are following healthy habits! Keep it up.")

    # ===============================
    # Feature Distribution Graph (Plotly)
    # ===============================
    st.subheader("🔎 Feature Distribution")
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    dist_feature = st.selectbox("Select a feature to analyze:", numeric_cols)

    if dist_feature in df_features.columns:
        fig = px.histogram(
            df_features, 
            x=dist_feature, 
            nbins=20, 
            opacity=0.7, 
            title=f"{dist_feature} Distribution",
            width=600, 
            height=400
        )

        # Add user value
        fig.add_vline(
            x=user_input.get(dist_feature), 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Your Value", 
            annotation_position="top"
        )

        # Animation style
        fig.update_traces(marker_color="skyblue", marker_line_width=1.2)
        fig.update_layout(
            bargap=0.05, 
            transition=dict(duration=800, easing="cubic-in-out")
        )

        st.plotly_chart(fig, use_container_width=True)
