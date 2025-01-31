#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
application_data = pd.read_csv('application_data.csv')

# Display basic information about the dataset
print("Application Data Info:")
print(application_data.info())

# Preview the dataset
print("\nApplication Data Preview:")
print(application_data.head())

# Print statistical summary
print("\nApplication Data Description:")
print(application_data.describe())

# Check for missing values
print("\nMissing Values in Application Data:")
print(application_data.isnull().sum())

# Fill missing numeric values with the median
numeric_columns = application_data.select_dtypes(include=['float64', 'int64']).columns
application_data[numeric_columns] = application_data[numeric_columns].fillna(application_data[numeric_columns].median())

# Fill missing categorical values with the mode
categorical_columns = application_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    application_data[col].fillna(application_data[col].mode()[0], inplace=True)

print("\nAfter handling missing values:")
print(application_data.isnull().sum())

# Remove duplicate rows
application_data = application_data.drop_duplicates()
print("\nAfter removing duplicates:")
print(application_data.info())

# Convert 'DAYS_BIRTH' to age in years
application_data['DAYS_BIRTH'] = application_data['DAYS_BIRTH'] / -365

print("\nAfter converting 'DAYS_BIRTH' to age:")
print(application_data[['DAYS_BIRTH']].head())

# Filter dataset before visualizations
application_data = application_data[(application_data['DAYS_BIRTH'] > 18) & (application_data['DAYS_BIRTH'] < 100)]

# Define correlation matrix
corr_matrix = application_data.corr()

# Filtering out 'XNA' values from 'CODE_GENDER' if they exist
defaulters = application_data[(application_data['TARGET'] == 1) & (application_data['CODE_GENDER'] != 'XNA')]
non_defaulters = application_data[(application_data['TARGET'] == 0) & (application_data['CODE_GENDER'] != 'XNA')]

# Plot Defaulters and Non-Defaulters by Gender
plt.figure(figsize=(20, 8))

# Defaulters by gender
plt.subplot(1, 2, 1)
sns.countplot(x='CODE_GENDER', data=defaulters, palette='Blues')
plt.title('Defaulters by Gender\n', fontsize=15, fontweight='bold', color='darkblue')
plt.xlabel('Gender')
plt.ylabel('Count')

# Non-defaulters by gender
plt.subplot(1, 2, 2)
sns.countplot(x='CODE_GENDER', data=non_defaulters, palette='Reds')
plt.title('Non-Defaulters by Gender\n', fontsize=15, fontweight='bold', color='darkred')
plt.xlabel('Gender')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Boxplot for Income, Credit, and Age
data1 = application_data['AMT_INCOME_TOTAL'].dropna()
data2 = application_data['AMT_CREDIT'].dropna()
data3 = application_data['DAYS_BIRTH'].dropna()

plt.figure(figsize=(8, 6))
plt.boxplot([data1, data2, data3], labels=['Income', 'Credit', 'Age'], notch=True, patch_artist=True)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Boxplot of Income, Credit, and Age')
plt.show()

# Plot distribution of loan amount
plt.figure(figsize=(8, 5))
plt.hist(application_data['AMT_CREDIT'], bins=30, color='green', edgecolor='black')
plt.title('Distribution of Loan Amount (AMT_CREDIT)')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()

# Correlation Bar Chart
corr_values = corr_matrix["TARGET"].drop("TARGET")
plt.figure(figsize=(10, 5))
corr_values.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Correlation of Features with Loan Default (TARGET)", fontsize=14)
plt.ylabel("Correlation Coefficient")
plt.xlabel("Features")
plt.axhline(0, color='black', linewidth=1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatter plot for Income vs. Credit with a Regression Line
plt.figure(figsize=(8, 5))
sns.regplot(x=application_data['AMT_INCOME_TOTAL'], y=application_data['AMT_CREDIT'], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.title('Income vs Loan Amount with Trend Line')
plt.xlabel('Total Income')
plt.ylabel('Loan Amount')
plt.show()
