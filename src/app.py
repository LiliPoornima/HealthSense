import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ===============================
# Unit Configuration
# ===============================
FEATURE_UNITS = {
    # Personal Information
    'age': 'years',
    'height': 'cm',
    'weight': 'kg',
    
    # Body Measurements
    'bmi': 'kg/m²',
    'waist_size': 'cm',
    'hip_size': 'cm',
    'chest_size': 'cm',
    
    # Vital Signs & Medical Measurements
    'blood_pressure': 'mmHg',
    'heart_rate': 'bpm',
    'temperature': '°C',
    'oxygen_saturation': '%',
    'glucose': 'mg/dL',
    'cholesterol': 'mg/dL',
    'hemoglobin': 'g/dL',
    'hba1c': '%',
    
    # Lifestyle Habits
    'sleep_hours': 'hours',
    'work_hours': 'hours',
    'daily_steps': 'steps',
    'meals_per_day': 'meals',
    'physical_activity': 'hours/week',
    'screen_time': 'hours/day',
    'caffeine_intake': 'cups/day',
    
    # Other measurements
    'systolic_bp': 'mmHg',
    'diastolic_bp': 'mmHg',
    'respiratory_rate': 'breaths/min',
    
    # Default fallback
    'default': ''
}

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
# Function to create Health Gauge Chart
# ===============================
def create_health_gauge(risk_percent):
    """Create a gauge chart showing health risk level"""
    
    # Determine color based on risk
    if risk_percent <= 30:
        color = "green"
        risk_label = "Low Risk"
    elif risk_percent <= 70:
        color = "orange"
        risk_label = "Moderate Risk"
    else:
        color = "red"
        risk_label = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_percent,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Health Risk Score<br><span style='font-size:0.8em;color:gray'>{risk_label}</span>", 
                 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.2)'},
                {'range': [30, 70], 'color': 'rgba(255, 165, 0, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.2)'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_percent}}))
    
    fig.update_layout(
        paper_bgcolor = "white",
        font = {'color': "darkblue", 'family': "Arial"},
        height=400
    )
    
    return fig

# ===============================
# Function to create Risk Trend Chart
# ===============================
def create_risk_trend_chart(history):
    """Create a line chart showing risk trend over time"""
    if len(history) < 2:
        return None
    
    # Extract data from history
    dates = [record['timestamp'] for record in history]
    risks = [record['probability'] * 100 if record['probability'] is not None else 0 
             for record in history]
    
    fig = go.Figure()
    
    # Add risk trend line
    fig.add_trace(go.Scatter(
        x=dates,
        y=risks,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=10, color='#636EFA', line=dict(color='white', width=2)),
        hovertemplate='<b>Date:</b> %{x}<br><b>Risk:</b> %{y:.1f}%<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=30, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold", annotation_position="right")
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  annotation_text="High Risk Threshold", annotation_position="right")
    
    fig.update_layout(
        title="Your Health Risk Trend Over Time",
        xaxis_title="Date",
        yaxis_title="Risk Score (%)",
        yaxis_range=[0, 100],
        hovermode='x unified',
        showlegend=False,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='white'
    )
    
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# ===============================
# Function to create Health Category Radar Chart
# ===============================
def create_health_radar_chart(user_input, df_features):
    """Create a radar chart comparing user metrics to population averages"""
    
    # Select key health categories
    categories_map = {
        'BMI': 'bmi',
        'Blood Pressure': 'blood_pressure',
        'Heart Rate': 'heart_rate',
        'Physical Activity': 'physical_activity',
        'Sleep Hours': 'sleep_hours'
    }
    
    user_values = []
    avg_values = []
    categories = []
    
    for category, col in categories_map.items():
        if col in user_input and col in df_features.columns:
            user_val = user_input.get(col)
            if isinstance(user_val, (int, float)):
                # Normalize to 0-100 scale
                min_val = df_features[col].min()
                max_val = df_features[col].max()
                avg_val = df_features[col].mean()
                
                user_normalized = ((user_val - min_val) / (max_val - min_val)) * 100
                avg_normalized = ((avg_val - min_val) / (max_val - min_val)) * 100
                
                user_values.append(user_normalized)
                avg_values.append(avg_normalized)
                categories.append(category)
    
    if not categories:
        return None
    
    fig = go.Figure()
    
    # Add user trace
    fig.add_trace(go.Scatterpolar(
        r=user_values + [user_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='Your Values',
        line=dict(color='#636EFA', width=2),
        fillcolor='rgba(99, 110, 250, 0.3)'
    ))
    
    # Add population average trace
    fig.add_trace(go.Scatterpolar(
        r=avg_values + [avg_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Population Average',
        line=dict(color='lightgray', width=2, dash='dash'),
        fillcolor='rgba(128, 128, 128, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False
            )
        ),
        showlegend=True,
        title="Health Metrics Comparison",
        height=450
    )
    
    return fig

# ===============================
# Function to create Feature Comparison Chart
# ===============================
def create_feature_comparison_chart(user_input, df_features):
    """Create a horizontal bar chart comparing user values to population stats"""
    
    # Select numeric features to compare
    features_to_compare = ['age', 'bmi', 'blood_pressure', 'heart_rate', 'sleep_hours']
    available_features = [f for f in features_to_compare if f in user_input and f in df_features.columns]
    
    if not available_features:
        return None
    
    comparison_data = []
    
    for feature in available_features:
        user_val = user_input.get(feature)
        if isinstance(user_val, (int, float)):
            pop_mean = df_features[feature].mean()
            pop_median = df_features[feature].median()
            
            comparison_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Your Value': user_val,
                'Population Mean': pop_mean,
                'Population Median': pop_median
            })
    
    if not comparison_data:
        return None
    
    df_compare = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    # Add bars for each metric
    fig.add_trace(go.Bar(
        name='Your Value',
        x=df_compare['Your Value'],
        y=df_compare['Feature'],
        orientation='h',
        marker=dict(color='#636EFA'),
        text=df_compare['Your Value'].round(1),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Population Mean',
        x=df_compare['Population Mean'],
        y=df_compare['Feature'],
        orientation='h',
        marker=dict(color='lightblue'),
        text=df_compare['Population Mean'].round(1),
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Population Median',
        x=df_compare['Population Median'],
        y=df_compare['Feature'],
        orientation='h',
        marker=dict(color='lightgray'),
        text=df_compare['Population Median'].round(1),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Your Values vs Population Statistics",
        xaxis_title="Value",
        yaxis_title="",
        barmode='group',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ===============================
# PDF Generation Function
# ===============================
def generate_pdf_report(result):
    """Generate a PDF report of the health state prediction"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    title = Paragraph("HealthSense - Health Prediction Report", title_style)
    story.append(title)
    
    # Date
    date_text = Paragraph(f"<i>Prediction Date: {result.get('timestamp', 'N/A')}</i>", styles['Normal'])
    story.append(date_text)
    story.append(Spacer(1, 20))
    
    # Prediction Results Section
    story.append(Paragraph("Prediction Results", heading_style))
    
    prediction = result["prediction"]
    probability = result["probability"]
    
    status = "🟥 Diseased" if prediction == 1 else "🟩 Healthy"
    status_text = Paragraph(f"<b>Status:</b> {status}", styles['Normal'])
    story.append(status_text)
    story.append(Spacer(1, 8))
    
    if probability is not None:
        risk_percent = probability * 100
        risk_text = Paragraph(f"<b>Risk Score:</b> {risk_percent:.1f}%", styles['Normal'])
        story.append(risk_text)
        story.append(Spacer(1, 8))
        
        if risk_percent <= 30:
            risk_level = "Low Risk - Health indicators suggest you are likely healthy"
        elif 30 < risk_percent <= 70:
            risk_level = "Moderate Risk - Some health indicators need attention"
        else:
            risk_level = "High Risk - Multiple health indicators suggest elevated disease risk"
        
        risk_level_text = Paragraph(f"<b>Risk Level:</b> {risk_level}", styles['Normal'])
        story.append(risk_level_text)
    
    story.append(Spacer(1, 20))
    
    # Health Recommendations Section
    story.append(Paragraph("Health Recommendations", heading_style))
    
    user_input = result["user_input"]
    recommendations = []
    
    # Generate recommendations
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
    
    alcohol = user_input.get("alcohol_consumption")
    if isinstance(alcohol, str) and alcohol.lower() not in ["none", "never", "low"]:
        recommendations.append("Alcohol Moderation: Consider limiting alcohol consumption to reduce long-term health risks.")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            rec_text = Paragraph(f"{i}. {rec}", styles['Normal'])
            story.append(rec_text)
            story.append(Spacer(1, 6))
    else:
        success_text = Paragraph("Great job! You're following healthy habits. Keep up the excellent work!", styles['Normal'])
        story.append(success_text)
    
    story.append(Spacer(1, 20))
    
    # Key Health Indicators Section
    story.append(Paragraph("Your Health Data Summary", heading_style))
    
    key_features = ['age', 'bmi', 'blood_pressure', 'heart_rate', 'smoking_status', 
                   'physical_activity', 'sleep_hours', 'alcohol_consumption']
    
    data = [['Feature', 'Your Value']]
    for feature in key_features:
        if feature in user_input:
            value = user_input[feature]
            label = feature.replace('_', ' ').title()
            unit = FEATURE_UNITS.get(feature, '')
            if isinstance(value, (int, float)):
                if unit:
                    data.append([f"{label} ({unit})", f"{value:.2f}"])
                else:
                    data.append([label, f"{value:.2f}"])
            else:
                data.append([label, str(value)])
    
    if len(data) > 1:
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        story.append(table)
    
    story.append(Spacer(1, 30))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=9,
        textColor=colors.grey
    )
    disclaimer_text = Paragraph(
        "<i>Disclaimer: This report is for informational purposes only and does not replace professional medical advice. Please consult with a healthcare provider for a comprehensive evaluation.</i>",
        disclaimer_style
    )
    story.append(disclaimer_text)
    
    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes

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
                
                # Show some key inputs with units
                st.markdown("**Key Health Indicators:**")
                key_features = ['age', 'bmi', 'blood_pressure', 'heart_rate', 'sleep_hours']
                display_data = {}

                for feature in key_features:
                    if feature in record['user_input']:
                        value = record['user_input'][feature]
                        unit = FEATURE_UNITS.get(feature, '')
                        label = feature.replace('_', ' ').title()
                        if unit:
                            display_data[f"{label} ({unit})"] = value
                        else:
                            display_data[label] = value

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
# PAGE NAVIGATION
# ===============================
elif st.session_state.current_page == "input":
    
    # Animation and header
    if lottie_animation:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(lottie_animation, height=200, key="health_anim")
    
    st.title("🩺 HealthSense - AI Health Risk Predictor")
    st.markdown("### Welcome! Get Your Personalized Health Prediction")
    
    st.info("👇 **Get Started:** Fill in your health details below and click the **Predict** button at the bottom to receive your personalized health risk assessment.")
    
    st.divider()
    
    # ===============================
    # UNIT CONVERSION HELPERS
    # ===============================
    def display_conversion_helpers():
        """Show unit conversion helpers for common measurements"""
        with st.expander("🔄 Unit Conversion Helpers", expanded=False):
            st.markdown("**Convert between different measurement systems:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Height Conversion**")
                feet = st.number_input("Feet", min_value=0, max_value=8, value=5, key="feet")
                inches = st.number_input("Inches", min_value=0, max_value=11, value=8, key="inches")
                total_inches = feet * 12 + inches
                cm = total_inches * 2.54
                st.success(f"**{feet}'{inches}\"** = **{cm:.1f} cm**")
            
            with col2:
                st.markdown("**Weight Conversion**")
                pounds = st.number_input("Pounds", min_value=0, max_value=500, value=150, key="pounds")
                kg = pounds * 0.453592
                st.success(f"**{pounds} lbs** = **{kg:.1f} kg**")
            
            with col3:
                st.markdown("**Temperature Conversion**")
                fahrenheit = st.number_input("°F", min_value=0, max_value=120, value=98, key="fahrenheit")
                celsius = (fahrenheit - 32) * 5/9
                st.success(f"**{fahrenheit}°F** = **{celsius:.1f}°C**")
    
    # Display the conversion helpers
    display_conversion_helpers()
    st.divider()
    
    # Input form on MAIN page (not sidebar)
    st.subheader("📋 Enter Your Health Information")

    # Define columns that should only take integers
    int_only_cols = [
        "survey_code", "age", "height", "waist_size",
        "blood_pressure", "heart_rate", "sleep_hours",
        "work_hours", "daily_steps", "meals_per_day", "physical_activity"
    ]
    
    # Create input fields for each category
    for category, columns in feature_categories.items():
        with st.expander(category, expanded=(category == "👤 Personal Information")):
            # Create columns for better layout
            num_cols = min(len(columns), 3)
            cols = st.columns(num_cols)
            
            for idx, col in enumerate(columns):
                with cols[idx % num_cols]:
                    # Get the unit for this feature
                    unit = FEATURE_UNITS.get(col, FEATURE_UNITS['default'])
                    col_label = col.replace('_', ' ').title()
                    
                    # Add unit to label if available
                    if unit:
                        display_label = f"{col_label} ({unit})"
                    else:
                        display_label = col_label
                    
                    if col in numeric_cols:
                        # Numeric input with units
                        if col in int_only_cols:
                            # Integer input
                            st.session_state.user_input[col] = int(st.number_input(
                                label=display_label,
                                min_value=int(df_features[col].min()),
                                max_value=int(df_features[col].max() * 1.5),
                                value=int(st.session_state.user_input.get(col, df_features[col].median())),
                                step=1,
                                format="%d",
                                key=f"input_{col}",
                                help=f"Range: {int(df_features[col].min())} - {int(df_features[col].max())} {unit}"
                            ))
                        else:
                            # Float input
                            st.session_state.user_input[col] = st.number_input(
                                label=display_label,
                                min_value=float(df_features[col].min()),
                                max_value=float(df_features[col].max() * 1.5),
                                value=float(st.session_state.user_input.get(col, df_features[col].median())),
                                step=0.01,
                                key=f"input_{col}",
                                help=f"Range: {df_features[col].min():.1f} - {df_features[col].max():.1f} {unit}"
                            )
                    else:
                        # Categorical input
                        options = df_features[col].dropna().unique().tolist()
                        default_value = st.session_state.user_input.get(col, options[0])
                        
                        if len(options) <= 2:
                            st.session_state.user_input[col] = st.radio(
                                display_label,
                                options,
                                index=options.index(default_value) if default_value in options else 0,
                                key=f"input_{col}"
                            )
                        else:
                            st.session_state.user_input[col] = st.selectbox(
                                display_label,
                                options,
                                index=options.index(default_value) if default_value in options else 0,
                                key=f"input_{col}"
                            )
    
    st.divider()
    
    # Predict button at the END of the page
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1E90FF; /* Dodger Blue */
        color: white;
    }
    div.stButton > button:first-child:hover {
        background-color: #104E8B; /* Dark Blue on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)
    st.markdown("### 🎯 Ready for Your Health Check?")
    col1, col2, col3 = st.columns([2, 1, 2])
   
    with col2:
        if st.button("🔍 Predict Health Risk", use_container_width=True):
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
# RESULTS PAGE
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
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 PDF Report",
            data=pdf_bytes,
            file_name=f"HealthSense_Report_{current_time}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    
    st.title("🩺 HealthSense - Your Health Results")
    st.caption(f"Assessment Date: {timestamp}")
    st.divider()
    
    # ===============================
    # Enhanced Visualization Section
    # ===============================
    st.subheader("📊 Visual Health Analysis")
    
    # Create columns for gauge and trend
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Health Gauge Chart
        if prediction_proba is not None:
            risk_percent = prediction_proba * 100
            gauge_fig = create_health_gauge(risk_percent)
            st.plotly_chart(gauge_fig, use_container_width=True)
    
    with viz_col2:
        # Risk Trend Chart (if history available)
        if len(st.session_state.prediction_history) >= 2:
            trend_fig = create_risk_trend_chart(st.session_state.prediction_history)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("📈 **Risk Trend Analysis**\n\nComplete more predictions to see your health trend over time!")
    
    st.divider()
    
    # ===============================
    # Health Metrics Comparison Section
    # ===============================
    st.subheader("🔍 Detailed Health Metrics")
    
    metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs([
        "📊 Radar Comparison", 
        "📈 Feature Analysis", 
        "📋 Distribution Comparison"
    ])
    
    with metrics_tab1:
        st.markdown("**Compare your health metrics with population averages:**")
        radar_fig = create_health_radar_chart(user_input, df_features)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient data for radar chart comparison.")
    
    with metrics_tab2:
        st.markdown("**See how your values compare to population statistics:**")
        comparison_fig = create_feature_comparison_chart(user_input, df_features)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        else:
            st.info("Insufficient data for feature comparison.")
    
    with metrics_tab3:
        st.markdown("**Analyze specific features in detail:**")
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
    
    st.divider()
    
    # ===============================
    # Main prediction result - VERTICAL LAYOUT
    # ===============================
    st.subheader("📊 Prediction Results")
    
    # VERTICAL LAYOUT - Status, Risk Level, Quick Stats in order
    with st.container():
        # Status Section
        st.markdown("### 🎯 Status")
        status_col1, status_col2 = st.columns([1, 2])
        
        with status_col1:
            if prediction == 1:
                st.error("**🟥 Diseased**")
            else:
                st.success("**🟩 Healthy**")
        
        with status_col2:
            if prediction == 1:
                st.info("""
                **Recommendation:** 
                Please consult with a healthcare professional for further evaluation and guidance.
                """)
            else:
                st.info("""
                **Great news!** 
                Your health indicators appear to be within normal ranges.
                """)
    
    st.divider()
    
    # Risk Level Section
    with st.container():
        st.markdown("### 📈 Risk Level")
        
        if prediction_proba is not None:
            risk_percent = prediction_proba * 100
            
            # Progress bar with color coding
            if risk_percent <= 30:
                progress_color = "green"
                risk_label = "Low Risk"
                risk_description = "Your health indicators suggest you are likely healthy."
            elif 30 < risk_percent <= 70:
                progress_color = "orange"
                risk_label = "Moderate Risk"
                risk_description = "Some health indicators need attention."
            else:
                progress_color = "red"
                risk_label = "High Risk"
                risk_description = "Multiple health indicators suggest elevated disease risk."
            
            # Display in columns for better layout
            risk_col1, risk_col2 = st.columns([1, 2])
            
            with risk_col1:
                # Risk metric
                st.metric(
                    label="Risk Score", 
                    value=f"{risk_percent:.1f}%",
                    delta=risk_label,
                    delta_color="off"
                )
                
                # Progress bar
                st.progress(int(risk_percent) / 100, text=f"{risk_percent:.1f}%")
            
            with risk_col2:
                # Risk interpretation
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 8px; background-color: #f8f9fa; border-left: 5px solid {progress_color}; margin-top: 10px;">
                <h4 style="margin: 0; color: {progress_color};">{risk_label}</h4>
                <p style="margin: 5px 0 0 0; font-size: 14px;">{risk_description}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Risk probability not available")
    
    st.divider()
    
    # Quick Stats Section
    with st.container():
        st.markdown("### 📊 Quick Stats")
        
        # Create 4 columns for stats
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            # Feature count
            total_features = len(user_input)
            st.metric(
                label="Features Analyzed", 
                value=total_features,
                help="Number of health indicators used in this Prediction"
            )
        
        with stat_col2:
            # Numeric metrics
            numeric_features = sum(1 for value in user_input.values() if isinstance(value, (int, float)))
            st.metric(
                label="Numeric Metrics", 
                value=numeric_features
            )
        
        with stat_col3:
            # Lifestyle factors
            categorical_features = total_features - numeric_features
            st.metric(
                label="Lifestyle Factors", 
                value=categorical_features
            )
        
        with stat_col4:
            # Data quality indicator
            completion_rate = (total_features / len(all_columns)) * 100
            st.metric(
                label="Data Completeness", 
                value=f"{completion_rate:.0f}%"
            )
    
    st.divider()
    
    # ===============================
    # Personalized Recommendations
    # ===============================
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
        # Display recommendations in columns for better layout
        rec_cols = st.columns(2)
        for idx, (title, message, level) in enumerate(recommendations):
            with rec_cols[idx % 2]:
                if level == "error":
                    st.error(f"**{title}:** {message}")
                elif level == "warning":
                    st.warning(f"**{title}:** {message}")
                else:
                    st.info(f"**{title}:** {message}")
    else:
        st.success("""
        🎉 **Excellent Health Habits!**
        
        You're maintaining healthy lifestyle choices across all major health indicators. 
        Continue with your current routine and regular health check-ups.
        """)
    
    st.divider()
    
    # ===============================
    # Your Input Summary Tab
    # ===============================
    st.subheader("📋 Your Health Data Summary")
    
    # Edit button moved to LEFT corner
    if st.button("✏️ Edit Data", type="primary"):
        st.session_state.current_page = "input"
        st.rerun()
    
    st.divider()
    
    for category, columns in feature_categories.items():
        with st.expander(category, expanded=False):
            for col in columns:
                value = user_input.get(col)
                col_label = col.replace('_', ' ').title()
                unit = FEATURE_UNITS.get(col, '')
                
                if unit:
                    display_text = f"**{col_label} ({unit}):**"
                else:
                    display_text = f"**{col_label}:**"
                
                if isinstance(value, (int, float)):
                    st.write(f"{display_text} {value:.2f}")
                else:
                    st.write(f"{display_text} {value}")
    
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

# ===============================
# DEFAULT PAGE HANDLER
# ===============================
else:
    # If page state is invalid, reset to input page
    st.session_state.current_page = "input"
    st.rerun()