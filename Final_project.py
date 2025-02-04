#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the datasets
application_data = pd.read_csv('application_data.csv')


# In[3]:


# Display the basic information of both datasets
print("Application Data Info:")
print(application_data.info())

# Preview the first few rows of each dataset
print("\nApplication Data Preview:")
print(application_data.head())

# printing the statistical summary
print("\nApplication Data Description:")
print(application_data.describe())



# Check for missing values in both datasets
print("\nMissing Values in Application Data:")
print(application_data.isnull().sum())


# In[4]:


# Calculate the number of missing values
miss_valappl = application_data.isnull().sum()

# Calculate the percentage of missing values
miss_valappl_percentage = (miss_valappl / len(application_data)) * 100


# In[5]:


# Fill missing numeric values with the median
numeric_columns = application_data.select_dtypes(include=['float64', 'int64']).columns
application_data[numeric_columns] = application_data[numeric_columns].fillna(application_data[numeric_columns].median())

# Fill missing categorical values with the mode
categorical_columns = application_data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    application_data[col].fillna(application_data[col].mode()[0], inplace=True)

print("\nAfter handling missing values:")
print(application_data.isnull().sum())


# In[6]:


# Remove duplicate rows
application_data = application_data.drop_duplicates()

print("\nAfter removing duplicates:")
print(application_data.info())


# In[7]:


# Convert 'DAYS_BIRTH' to age in years
application_data['DAYS_BIRTH'] = application_data['DAYS_BIRTH'] / -365  # Convert to positive age

print("\nAfter converting 'DAYS_BIRTH' to age:")
print(application_data[['DAYS_BIRTH']].head())


# In[8]:


# Final review of the cleaned dataset
print("\nFinal cleaned data info:")
print(application_data.info())

print("\nFinal cleaned dataset preview:")
print(application_data.head())


# In[9]:


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


# In[10]:


# Filter the dataset before plotting
application_data = application_data[(application_data['DAYS_BIRTH'] > 18) & (application_data['DAYS_BIRTH'] < 100)]

# Plot distribution of loan amount
plt.figure(figsize=(8, 5))
plt.hist(application_data['AMT_CREDIT'], bins=30, color='green', edgecolor='black')

plt.title('Distribution of Loan Amount (AMT_CREDIT)')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()


# In[22]:


# Simulating application_data with random correlations for demonstration
np.random.seed(42)
num_features = 30  # Reducing number of features for readability
data = np.random.rand(100, num_features)
column_names = [f'Feature_{i}' for i in range(num_features)]
application_data = pd.DataFrame(data, columns=column_names)

# Compute the correlation matrix
corr_matrix = application_data.corr()

# Increase figure size for better visibility
plt.figure(figsize=(20, 15))

# First heatmap without annotations (to get an overview)
sns.heatmap(corr_matrix, cmap="coolwarm", linewidths=0.5, annot=False)

# Filter correlations that are strong (above 0.3 or below -0.3) for focused visualization
filtered_corr = corr_matrix[(corr_matrix >= 0.3) | (corr_matrix <= -0.3)]
plt.figure(figsize=(20, 15))
sns.heatmap(filtered_corr, cmap="coolwarm", linewidths=0.5, annot=True, fmt=".2f")

# Adjust labels for clarity
plt.xticks(rotation=90)  # Rotate x-axis labels
plt.yticks(rotation=0)   # Keep y-axis labels horizontal

plt.title("Feature Correlation Heatmap", fontsize=16)
plt.show()


# In[12]:


# Scatter plot for the application_data dataset

# Assuming 'AMT_INCOME_TOTAL' and 'AMT_CREDIT' are columns in the application_data dataframe
x = application_data['AMT_INCOME_TOTAL']
y = application_data['AMT_CREDIT']

# Create a scatter plot
plt.scatter(x, y, label='Income vs Credit', color='green', marker='o')

# Add title and labels
plt.title('Scatter Plot of Income vs Credit')
plt.xlabel('Total Income')
plt.ylabel('Credit Amount')

# Show the legend
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()


# In[13]:


# Select three numerical columns from application_data
data1 = application_data['AMT_INCOME_TOTAL'].dropna()
data2 = application_data['AMT_CREDIT'].dropna()
data3 = application_data['DAYS_BIRTH'].dropna()

# Combine the data for the box plot
data = [data1, data2, data3]

# Create a box plot
plt.boxplot(data, labels=['Income', 'Credit', 'Days of Birth'], notch=True, vert=True, patch_artist=True)

# Add title and labels
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Boxplot of Income, Credit, and Days of Birth')

# Display the plot
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




