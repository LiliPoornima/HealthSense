# 🩺 HealthSense - AI-Powered Health Risk Predictor



**HealthSense** is an intelligent web-based application that leverages machine learning to predict health risks based on lifestyle and demographic factors. Using advanced AI algorithms, it provides personalized health insights, risk assessments, and actionable recommendations to help users make informed decisions about their health and wellness.

The home page highlights HealthSense’s mission — empowering users through AI-driven predictions, interactive visualizations, and secure, real-time health assessments. Users can easily input their data, receive instant risk evaluations, and track their health progress over time through intuitive charts and dashboards.

## Features

###  **Core Functionality**
- **AI-Powered Predictions**: Uses trained Decision Tree model for accurate health risk assessment
- **Personalized Risk Scoring**: Provides percentage-based risk scores with detailed explanations
- **Interactive Web Interface**: User-friendly Streamlit-based dashboard
- **Real-time Analysis**: Instant predictions with comprehensive health insights

###  **Advanced Analytics**
- **Visual Health Dashboard**: Interactive gauge charts, trend analysis, and risk visualization
- **Population Comparison**: Compare your health metrics against population averages
- **Radar Charts**: Multi-dimensional health metric comparison
- **Distribution Analysis**: See where you stand in population health distributions
- **Trend Tracking**: Monitor your health risk changes over time

###  **Personalized Insights**
- **Health Recommendations**: Tailored advice based on your specific risk factors
- **Risk Category Classification**: Low, Moderate, and High-risk categorization
- **Feature Analysis**: Detailed breakdown of factors affecting your health score
- **Actionable Guidance**: Specific steps to improve your health outcomes

###  **User Experience**
- **Unit Conversion Tools**: Easy conversion between metric and imperial units
- **Prediction History**: Track your health assessments over time
- **PDF Report Generation**: Download comprehensive health reports
- **Responsive Design**: Works seamlessly on desktop and mobile devices

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LiliPoornima/HealthSense.git
   cd HealthSense
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/app.py
   ```

4. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## Project Structure

```
HealthSense/
│
├──  data/
│   ├── raw/                          # Original dataset
│   │   └── health_lifestyle_classification.csv
│   └── processed/                    # Preprocessed data
│       └── preprocessed_dataset.csv
│
├──  models/                        # Trained models and artifacts
│   ├── best_model_Decision_Tree.pkl  # Trained Decision Tree model
│   ├── feature_names.pkl            # Feature names for model input
│   └── scaler.pkl                   # Feature scaler for preprocessing
│
├──  src/                          # Source code
│   ├── app.py                       # Main Streamlit application
│   ├── eda.py                       # Exploratory Data Analysis
│   ├── preprocessing.py             # Data preprocessing pipeline
│   └── train_model.py              # Model training and evaluation
│
├──  requirements.txt              # Python dependencies
└──  README.md                     # Project documentation
```

## Usage Guide

### 1. **Health Data Input**
- Fill in your personal information (age, height, weight, etc.)
- Enter vital signs (blood pressure, heart rate, temperature)
- Provide lifestyle information (exercise, sleep, diet habits)
- Include medical history and environmental factors

### 2. **Get Your Prediction**
- Click the "🔍 Predict Health Risk" button
- View your risk score and classification
- Explore detailed analytics and visualizations

### 3. **Analyze Results**
- **Visual Dashboard**: Interactive charts showing your health status
- **Risk Breakdown**: Understand what factors contribute to your risk
- **Recommendations**: Get personalized health improvement suggestions
- **Comparisons**: See how you compare to population averages

### 4. **Track Progress**
- Use the prediction history to monitor changes over time
- Download PDF reports for your records
- Track improvements in your health metrics

## Model Information

### **Algorithm**: Decision Tree Classifier
- **Accuracy**: Optimized for health risk prediction
- **Features**: 20+ lifestyle and demographic factors
- **Training**: Balanced dataset with SMOTE for handling class imbalance
- **Validation**: Cross-validated with multiple evaluation metrics

### **Key Features Used**:
- **Demographics**: Age, gender, BMI, body measurements
- **Vital Signs**: Blood pressure, heart rate, temperature
- **Lifestyle**: Physical activity, sleep patterns, diet habits
- **Health Behaviors**: Smoking, alcohol consumption, stress levels
- **Medical History**: Family history, existing conditions

## Data Pipeline

### 1. **Data Collection**
- Health and lifestyle classification dataset
- Comprehensive feature engineering
- Quality validation and cleaning

### 2. **Preprocessing**
```python
# Data cleaning and preprocessing steps
- Handle missing values
- Feature scaling and normalization
- Categorical encoding
- Class balancing with SMOTE
```

### 3. **Model Training**
```python
# Multiple algorithms compared:
- Logistic Regression
- Decision Tree (Best performing)
- Random Forest
- XGBoost
```

### 4. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC analysis
- Cross-validation testing
- Feature importance analysis

## User Interface

### **Main Dashboard**
- Clean, intuitive design with health-focused color scheme
- Progressive disclosure of information
- Interactive elements with real-time feedback

### **Visualization Components**
- **Health Gauge**: Risk level visualization
- **Trend Charts**: Historical risk tracking
- **Radar Charts**: Multi-metric comparison
- **Distribution Plots**: Population-based analysis

### **Mobile Responsiveness**
- Optimized for all screen sizes
- Touch-friendly interface
- Consistent experience across devices

## Privacy & Security

- **No Data Storage**: Personal health information is not stored permanently
- **Session-Based**: Data exists only during your session
- **Local Processing**: All calculations performed locally
- **Privacy-First**: No personal data transmitted to external servers

##  Contributing

We welcome contributions to improve HealthSense! Here's how you can help:

### **Areas for Contribution**
- Model improvements and new algorithms
- Additional health features and metrics
- UI/UX enhancements
- Documentation improvements
- Bug fixes and performance optimizations

### **Getting Started**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap

### **Model Enhancements**
- [ ] **Fine Tuning the models**: Continuous model improvement

## Known Issues

- Large datasets may require additional processing time
- Some categorical features may need manual encoding adjustments



## Performance Metrics
```
Model Performance:
├── Accuracy: 85%+
├── Precision: 69%+
├── Recall: 67%+
├── F1-Score: 67%+
└── ROC-AUC: 0.49+
```


---

**⚠️ Medical Disclaimer**: HealthSense is designed for informational and educational purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions and never disregard professional medical advice based on information from this application.

---



