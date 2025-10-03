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

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]
print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
if duplicate_rows.shape[0] > 0:
    print("Duplicate rows found:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")
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


# Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude non-feature numeric columns if needed (like 'target')
if 'target' in numeric_cols:
    numeric_cols.remove('target')

plt.figure(figsize=(20,20))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()