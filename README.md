# Earthquake-Prediction-Model
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load earthquake data (You would need a real earthquake dataset)
# For simplicity, let's assume you have a CSV file with columns: 'latitude', 'longitude', 'depth', 'magnitude'
data = pd.read_csv('earthquake_data.csv')

# Prepare features and target variable
X = data[['latitude', 'longitude', 'depth']]
y = data['magnitude']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
