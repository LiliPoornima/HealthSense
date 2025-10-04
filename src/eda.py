# src/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/health_lifestyle_classification.csv")

print("Number of rows and columns in the dataset:")
print(df.shape)
print()

print("First 5 rows:")
print(df.head())
print()

# Check types & missing values
print("Info about dataset:")
print(df.info())
print()

print("Missing values in each column:")
print(df.isnull().sum())
print()

# Class balance
print("Class balance (normalized):")
print(df['target'].value_counts(normalize=True))
print()

# Numeric summary
print("Numeric summary:")
print(df.describe())
print()

# Plot class distribution
sns.countplot(x="target", data=df)
plt.title("Class Distribution")
plt.show()
