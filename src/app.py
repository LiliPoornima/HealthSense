import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
from fpdf import FPDF
import base64

# ===============================
# Helper function to load Lottie animations
# ===============================
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animation
lottie_url = "https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_animation = load_lottieurl(lottie_url)

# ===============================
# PDF Generation Function
# ===============================
def generate_pdf_report(result):
    """Generate a PDF report of the health assessment"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "HealthSense - Health Risk Assessment Report", ln=True, align="C")
    pdf.ln(5)
    
    # Date
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Assessment Date: {result.get('timestamp', 'N/A')}", ln=True, align="C")
    pdf.ln(10)
    
    # Prediction Result
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Assessment Results", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    prediction = result["prediction"]
    probability = result["probability"]
    
    status = "Diseased" if prediction == 1 else "Healthy"
    pdf.cell(0, 10, f"Status: {status}", ln=True)
    
    if probability is not None:
        risk_percent = probability * 100
        pdf.cell(0, 10, f"Risk Score: {risk_percent:.1f}%", ln=True)
        
        if risk_percent <= 30:
            risk_level = "Low Risk - Health indicators suggest you are likely healthy"
        elif 30 < risk_percent <= 70:
            risk_level = "Moderate Risk - Some health indicators need attention"
        else:
            risk_level = "High Risk - Multiple health indicators suggest elevated disease risk"
        
        pdf.multi_cell(0, 10, f"Risk Level: {risk_level}")
    
    pdf.ln(10)
    
    # Health Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Health Recommendations", ln=True)
    pdf.ln(5)
    
    user_input = result["user_input"]
    recommendations = []
    
    # Generate recommendations (same logic as in the app)
    bmi = user_input.get("bmi")
    if isinstance(bmi, (int, float)) and bmi > 25:
        recommendations.append("Weight Management: Your BMI is above the healthy range. Consider a balanced diet and regular exercise.")
    
    bp = user_input.get("blood_pressure")
    if isinstance(bp, (int, float)) and bp > 120:
        recommendations.append("Blood Pressure: Your blood pressure is elevated. Reduce salt intake and manage stress.")
    
    exercise = user_input.get("physical_activity")
    if isinstance(exercise, str) and exercise.lower() in ["low", "sedentary"]:
        recommendations.append("Physical Activity: Increase your activity. Aim for 30 minutes of exercise daily.")
    
    smoking = user_input.get("smoking_status")
    if isinstance(smoking, str) and smoking.lower() not in ["non-smoker", "never"]:
        recommendations.append("Smoking Cessation: Quitting smoking is crucial for your health.")
    
    if recommendations:
        pdf.set_font("Arial", "", 11)
        for i, rec in enumerate(recommendations, 1):
            pdf.multi_cell(0, 8, f"{i}. {rec}")
            pdf.ln(2)
    else:
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 10, "Great job! You're following healthy habits. Keep up the excellent work!")
    
    pdf.ln(10)
    
    # Key Health Indicators
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Your Health Data Summary", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 10)
    
    # Display key metrics
    key_features = ['age', 'bmi', 'blood_pressure', 'heart_rate', 'smoking_status', 
                   'physical_activity', 'sleep_hours', 'alcohol_consumption']
    
    for feature in key_features:
        if feature in user_input:
            value = user_input[feature]
            label = feature.replace('_', ' ').title()
            if isinstance(value, (int, float)):
                pdf.cell(0, 8, f"{label}: {value:.2f}", ln=True)
            else:
                pdf.cell(0, 8, f"{label}: {value}", ln=True)
    
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(0, 6, "Disclaimer: This assessment is for informational purposes only and does not replace professional medical advice. Please consult with a healthcare provider for a comprehensive evaluation.")
    
    return pdf.output(dest='S').encode('latin-1')

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
st.set_page_config(page_title="HealthSense - AI Health Risk Predictor", page_icon="🩺", layout="wide")

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(os.path.join(DATA_DIR, "preprocessed_dataset.csv"))
df_features = df.drop(columns=["target"], errors="ignore")

numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

# ===============================
# Smart Categorization Function
# ===============================
def categorize_feature(col_name):
    """Intelligently categorize features into user-friendly groups"""
    col_lower = col_name.lower()
    
    # Personal Information
    if any(keyword in col_lower for keyword in ['age', 'gender', 'sex', 'height', 'weight', 'device', 'supplement', 'dosage']):
        return "👤 Personal Information"
    
    # Body Measurements
    elif any(keyword in col_lower for keyword in ['bmi', 'waist', 'hip', 'chest', 'body', 'size', 'estimated', 'scaled', 'corrected']):
        return "📏 Body Measurements"
    
    # Vital Signs & Medical Measurements
    elif any(keyword in col_lower for keyword in ['blood_pressure', 'bp', 'heart_rate', 'pulse', 
                                                     'temperature', 'oxygen', 'glucose', 'cholesterol',
                                                     'sugar', 'hemoglobin', 'hba1c']):
        return "🫀 Vital Signs & Medical Measurements"
    
    # Lifestyle Habits
    elif any(keyword in col_lower for keyword in ['smoking', 'alcohol', 'caffeine', 'sleep', 
                                                     'exercise', 'physical_activity', 'diet',
                                                     'sunlight', 'screen_time']):
        return "🏃 Lifestyle Habits"
    
    # Mental & Emotional Health
    elif any(keyword in col_lower for keyword in ['stress', 'anxiety', 'depression', 'mental',
                                                     'mood', 'psychological']):
        return "🧠 Mental & Emotional Health"
    
    # Medical History
    elif any(keyword in col_lower for keyword in ['family_history', 'medical_history', 'disease',
                                                     'diabetes', 'hypertension', 'cancer', 'asthma',
                                                     'allergy', 'medication', 'prescription']):
        return "📋 Medical History"
    
    # Environmental & Social Factors
    elif any(keyword in col_lower for keyword in ['pollution', 'environment', 'occupation',
                                                     'income', 'education', 'social', 'living']):
        return "🌍 Environmental & Social Factors"
    
    # Other/General
    else:
        return "📊 Other Health Indicators"

# ===============================
# Organize features by category
# ===============================
all_columns = numeric_cols + categorical_cols
feature_categories = {}

for col in all_columns:
    category = categorize_feature(col)
    if category not in feature_categories:
        feature_categories[category] = []
    feature_categories[category].append(col)

# Sort categories in a logical order
category_order = [
    "👤 Personal Information",
    "📏 Body Measurements",
    "🫀 Vital Signs & Medical Measurements",
    "🏃 Lifestyle Habits",
    "🧠 Mental & Emotional Health",
    "📋 Medical History",
    "🌍 Environmental & Social Factors",
    "📊 Other Health Indicators"
]

# Keep only categories that have features
feature_categories = {cat: feature_categories[cat] for cat in category_order if cat in feature_categories}

# ===============================
# Initialize session state
# ===============================
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "current_page" not in st.session_state:
    st.session_state.current_page = "input"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
if "show_history" not in st.session_state:
    st.session_state.show_history = False

# ===============================
# SIDEBAR - App Info and History
# ===============================
st.sidebar.title("🩺 HealthSense")
st.sidebar.markdown("### *AI-Powered Health Analytics*")
st.sidebar.divider()

# Navigation info
st.sidebar.subheader("📍 Current Page")
if st.session_state.current_page == "input":
    st.sidebar.info("📝 **Input Page**")
    st.sidebar.caption("Fill in your health details")
else:
    st.sidebar.success("✅ **Results Page**")
    st.sidebar.caption("View your assessment")

st.sidebar.divider()

# History section
st.sidebar.subheader("📜 Prediction History")
st.sidebar.caption(f"Total Predictions: {len(st.session_state.prediction_history)}")

if st.sidebar.button("📊 View History", use_container_width=True):
    st.session_state.show_history = True

if st.sidebar.button("🏠 Back to Main", use_container_width=True):
    st.session_state.show_history = False
    st.session_state.current_page = "input"
    st.rerun()

st.sidebar.divider()

# App Information
st.sidebar.subheader("ℹ️ About HealthSense")
st.sidebar.markdown("""
**HealthSense** uses advanced AI to analyze your health data and provide personalized risk insights.

**Features:**
- 🎯 AI-powered predictions
- 📊 Comprehensive analytics
- 💡 Personalized insights
- 📜 Historical tracking
""")

st.sidebar.divider()
st.sidebar.caption("⚠️ **Disclaimer:** This tool is for informational purposes only and does not replace professional medical advice.")

# ===============================
# HISTORY PAGE
# ===============================
if st.session_state.show_history:
    st.title("📜 Prediction History - HealthSense")
    st.markdown("### Track Your Health Journey Over Time")
    st.divider()
    
    if len(st.session_state.prediction_history) == 0:
        st.info("📭 No previous predictions available. Complete your first analysis to begin tracking your health data.")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("➕ New Prediction", use_container_width=True):
                st.session_state.show_history = False
                st.session_state.current_page = "input"
                st.rerun()
    else:
        # Display history in reverse chronological order
        for idx, record in enumerate(reversed(st.session_state.prediction_history)):
            with st.expander(f"🔍 Prediction #{len(st.session_state.prediction_history) - idx} - {record['timestamp']}", expanded=(idx == 0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Status**")
                    if record['prediction'] == 1:
                        st.error("🟥 Diseased")
                    else:
                        st.success("🟩 Healthy")
                
                with col2:
                    st.markdown("**Risk Score**")
                    if record['probability'] is not None:
                        st.metric("", f"{record['probability'] * 100:.1f}%")
                
                with col3:
                    st.markdown("**Features**")
                    st.metric("", len(record['user_input']))
                
                st.divider()
                
                # Show some key inputs
                st.markdown("**Key Health Indicators:**")
                key_features = ['age', 'bmi', 'blood_pressure', 'heart_rate', 'smoking_status', 'physical_activity']
                display_data = {}
                
                for feature in key_features:
                    if feature in record['user_input']:
                        value = record['user_input'][feature]
                        display_data[feature.replace('_', ' ').title()] = value
                
                if display_data:
                    cols = st.columns(len(display_data))
                    for idx, (key, value) in enumerate(display_data.items()):
                        with cols[idx]:
                            if isinstance(value, (int, float)):
                                st.metric(key, f"{value:.1f}")
                            else:
                                st.metric(key, value)
        
        st.divider()
        
        # Clear history button
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

# ===============================
# PAGE 1: INPUT FORM (Main Page)
# ===============================
elif st.session_state.current_page == "input":
    
    # Animation and header
    if lottie_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_animation, height=200, key="health_anim")
    
    st.title("🩺 HealthSense - AI Health Risk Predictor")
    st.markdown("### Welcome! Get Your Personalized Health Assessment")
    
    st.info("👇 **Get Started:** Fill in your health details below and click the **Predict** button at the bottom to receive your personalized health risk assessment.")
    
    st.divider()
    
    # Input form on MAIN page (not sidebar)
    st.subheader("📋 Enter Your Health Information")
    
    # Create input fields for each category
    for category, columns in feature_categories.items():
        with st.expander(category, expanded=(category == "👤 Personal Information")):
            # Create columns for better layout
            num_cols = min(len(columns), 3)
            cols = st.columns(num_cols)
            
            for idx, col in enumerate(columns):
                with cols[idx % num_cols]:
                    if col in numeric_cols:
                        # Numeric input
                        col_label = col.replace('_', ' ').title()
                        st.session_state.user_input[col] = st.number_input(
                            label=col_label,
                            min_value=float(df_features[col].min()),
                            max_value=float(df_features[col].max() * 1.5),
                            value=st.session_state.user_input.get(col, float(df_features[col].median())),
                            step=0.01,
                            key=f"input_{col}",
                            help=f"Range: {df_features[col].min():.1f} - {df_features[col].max():.1f}"
                        )
                    else:
                        # Categorical input
                        col_label = col.replace('_', ' ').title()
                        options = df_features[col].dropna().unique().tolist()
                        default_value = st.session_state.user_input.get(col, options[0])
                        
                        if len(options) <= 2:
                            st.session_state.user_input[col] = st.radio(
                                col_label,
                                options,
                                index=options.index(default_value) if default_value in options else 0,
                                key=f"input_{col}"
                            )
                        else:
                            st.session_state.user_input[col] = st.selectbox(
                                col_label,
                                options,
                                index=options.index(default_value) if default_value in options else 0,
                                key=f"input_{col}"
                            )
    
    st.divider()
    
    # Predict button at the END of the page
    st.markdown("### 🎯 Ready for Your Assessment?")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔍 Predict Health Risk", type="primary", use_container_width=True):
            # Perform prediction
            user_input = st.session_state.user_input.copy()
            input_df = pd.DataFrame([user_input])

            # Handle numeric features
            scaler_features = scaler.feature_names_in_
            for col in scaler_features:
                if col not in input_df.columns:
                    if col in df_features.columns:
                        input_df[col] = df_features[col].median()
                    else:
                        input_df[col] = 0
            input_df[scaler_features] = scaler.transform(input_df[scaler_features])

            # One-hot encode categorical safely
            for col in categorical_cols:
                for feature in feature_names:
                    if feature.startswith(col + "_"):
                        cat_value = feature.replace(col + "_", "")
                        if col in input_df.columns:
                            input_df[feature] = (input_df[col] == cat_value).astype(int)
                        else:
                            input_df[feature] = 0
                if col in input_df.columns:
                    input_df.drop(columns=[col], inplace=True)

            # Reindex to match model
            input_df = input_df.reindex(columns=feature_names, fill_value=0)

            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

            # Store results
            st.session_state.prediction_result = {
                "prediction": prediction,
                "probability": prediction_proba,
                "user_input": user_input,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to history
            st.session_state.prediction_history.append(st.session_state.prediction_result.copy())
            
            # Navigate to results page
            st.session_state.current_page = "results"
            st.rerun()

# ===============================
# PAGE 2: RESULTS
# ===============================
elif st.session_state.current_page == "results":
    
    result = st.session_state.prediction_result
    prediction = result["prediction"]
    prediction_proba = result["probability"]
    user_input = result["user_input"]
    timestamp = result.get("timestamp", "N/A")
    
    # PDF Download Button in Top Right Corner
    col1, col2 = st.columns([5, 1])
    with col2:
        pdf_bytes = generate_pdf_report(result)
        st.download_button(
            label="📥 PDF",
            data=pdf_bytes,
            file_name=f"HealthSense_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    
    st.title("🩺 HealthSense - Your Health Risk Assessment")
    st.caption(f"Assessment Date: {timestamp}")
    st.divider()
    
    # Main prediction result
    st.subheader("📊 Prediction Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### Status")
        if prediction == 1:
            st.error("### 🟥 Diseased")
        else:
            st.success("### 🟩 Healthy")
    
    with col2:
        if prediction_proba is not None:
            risk_percent = prediction_proba * 100
            st.markdown("### Risk Level")
            st.progress(int(risk_percent) / 100)
            st.metric("Risk Score", f"{risk_percent:.1f}%")
            
            if risk_percent <= 30:
                st.success("✅ **Low Risk:** Your health indicators suggest you are likely healthy.")
            elif 30 < risk_percent <= 70:
                st.warning("⚠️ **Moderate Risk:** Some health indicators need attention.")
            else:
                st.error("🚨 **High Risk:** Multiple health indicators suggest elevated disease risk.")
    
    with col3:
        st.markdown("### Quick Stats")
        total_features = len(user_input)
        st.metric("Features Analyzed", total_features)
    
    st.divider()
    
    # Personalized Recommendations
    st.subheader("💡 Personalized Health Recommendations")
    
    recommendations = []
    
    # Check various health indicators
    bmi = user_input.get("bmi")
    if isinstance(bmi, (int, float)) and bmi > 25:
        recommendations.append(("⚖️ Weight Management", "Your BMI is above the healthy range. Consider a balanced diet and regular exercise to maintain a healthy weight.", "warning"))
    
    bp = user_input.get("blood_pressure")
    if isinstance(bp, (int, float)) and bp > 120:
        recommendations.append(("🫀 Blood Pressure", "Your blood pressure is elevated. Reduce salt intake, manage stress, and consult a healthcare provider.", "warning"))
    
    exercise = user_input.get("physical_activity")
    if isinstance(exercise, str) and exercise.lower() in ["low", "sedentary"]:
        recommendations.append(("🏃 Physical Activity", "Increase your physical activity. Aim for at least 30 minutes of moderate exercise daily.", "info"))
    
    smoking = user_input.get("smoking_status")
    if isinstance(smoking, str) and smoking.lower() not in ["non-smoker", "never"]:
        recommendations.append(("🚭 Smoking Cessation", "Quitting smoking is one of the best things you can do for your health. Seek support if needed.", "error"))
    
    alcohol = user_input.get("alcohol_consumption")
    if isinstance(alcohol, str) and alcohol.lower() not in ["none", "never", "low"]:
        recommendations.append(("🍷 Alcohol Moderation", "Consider limiting alcohol consumption to reduce long-term health risks.", "warning"))
    
    caffeine = user_input.get("caffeine_intake")
    if isinstance(caffeine, str) and caffeine.lower() in ["high", "very high"]:
        recommendations.append(("☕ Caffeine Intake", "High caffeine consumption may affect sleep and anxiety. Consider moderating your intake.", "info"))
    
    sunlight = user_input.get("sunlight_exposure")
    if isinstance(sunlight, str) and sunlight.lower() in ["low", "minimal"]:
        recommendations.append(("🌞 Vitamin D", "Increase sunlight exposure for better Vitamin D levels. Aim for 15-20 minutes of sun daily.", "info"))
    
    sleep = user_input.get("sleep_hours") if "sleep_hours" in user_input else user_input.get("sleep")
    if sleep is not None:
        if isinstance(sleep, (int, float)) and sleep < 7:
            recommendations.append(("😴 Sleep Quality", "Aim for 7-9 hours of quality sleep per night for optimal health.", "warning"))
    
    if recommendations:
        for title, message, level in recommendations:
            if level == "error":
                st.error(f"**{title}:** {message}")
            elif level == "warning":
                st.warning(f"**{title}:** {message}")
            else:
                st.info(f"**{title}:** {message}")
    else:
        st.success("🎉 **Great Job!** You're following healthy habits. Keep up the excellent work!")
    
    st.divider()
    
    # Feature Analysis Section
    st.subheader("🔎 Detailed Feature Analysis")
    
    tab1, tab2 = st.tabs(["📊 Distribution Comparison", "📋 Your Input Summary"])
    
    with tab1:
        st.markdown("**Compare your values with the overall population distribution:**")
        dist_feature = st.selectbox(
            "Select a feature to analyze:",
            numeric_cols,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if dist_feature in df_features.columns:
            fig = px.histogram(
                df_features, 
                x=dist_feature, 
                nbins=30, 
                opacity=0.7, 
                title=f"{dist_feature.replace('_', ' ').title()} - Population Distribution",
                labels={dist_feature: dist_feature.replace('_', ' ').title()},
                color_discrete_sequence=['#636EFA']
            )
            
            # Add user value
            user_value = user_input.get(dist_feature)
            if user_value is not None:
                try:
                    user_value = float(user_value)
                    fig.add_vline(
                        x=user_value,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text="Your Value",
                        annotation_position="top right"
                    )
                except ValueError:
                    pass
            
            fig.update_layout(
                bargap=0.05,
                showlegend=False,
                height=450,
                xaxis_title=dist_feature.replace('_', ' ').title(),
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Your Value", f"{user_value:.2f}" if isinstance(user_value, (int, float)) else str(user_value))
            with col2:
                st.metric("Population Mean", f"{df_features[dist_feature].mean():.2f}")
            with col3:
                st.metric("Population Median", f"{df_features[dist_feature].median():.2f}")
            with col4:
                st.metric("Population Std Dev", f"{df_features[dist_feature].std():.2f}")
    
    with tab2:
        st.markdown("**Review all the information you provided:**")
        
        # Edit button at TOP of summary section
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("✏️ Edit Data", type="primary", use_container_width=True):
                st.session_state.current_page = "input"
                st.rerun()
        
        st.divider()
        
        for category, columns in feature_categories.items():
            with st.expander(category, expanded=False):
                for col in columns:
                    value = user_input.get(col)
                    col_label = col.replace('_', ' ').title()
                    if isinstance(value, (int, float)):
                        st.write(f"**{col_label}:** {value:.2f}")
                    else:
                        st.write(f"**{col_label}:** {value}")
    
    st.divider()
    
    # Call to action
    st.info("💡 **Next Steps:** Consult with a healthcare professional for a comprehensive evaluation. This tool provides predictions based on data patterns and should not replace medical advice.")
    
    # New prediction button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🔄 New Prediction", use_container_width=True):
            st.session_state.current_page = "input"
            st.session_state.prediction_result = None
            st.rerun()