# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the Dataset
# Assuming a CSV file named 'house_prices.csv' containing data
# Replace the path with your actual dataset path
data = pd.read_csv('house_prices.csv')

# 2. Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

print("\nSummary statistics:")
print(data.describe())

print("\nChecking for missing values:")
print(data.isnull().sum())

# 3. Data Cleaning
# For simplicity, let's drop rows with missing values
data_cleaned = data.dropna()

# 4. Feature Engineering
# Let's create a new feature: Price per square foot
data_cleaned['price_per_sqft'] = data_cleaned['Price'] / data_cleaned['Size']

# Encode categorical variables (if any) using one-hot encoding
data_encoded = pd.get_dummies(data_cleaned, drop_first=True)

# 5. Exploratory Data Analysis (EDA)
# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Distribution of house prices
plt.figure(figsize=(6, 4))
sns.histplot(data_cleaned['Price'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.show()

# Scatter plot of Size vs. Price
plt.figure(figsize=(6, 4))
plt.scatter(data_cleaned['Size'], data_cleaned['Price'])
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Size vs Price')
plt.show()

# 6. Preparing Data for Modeling
X = data_encoded.drop(columns=['Price'])
y = data_encoded['Price']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Model Building
# Use a simple Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 8. Model Evaluation
# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# 9. Plotting Actual vs Predicted Prices
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

