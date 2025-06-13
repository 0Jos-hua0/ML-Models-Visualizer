import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Example dataset: House Size (in square feet), Number of Bedrooms, and House Price
data = {
    'House Size (sqft)': [800, 900, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
    'Bedrooms': [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'Price': [150000, 180000, 200000, 250000, 300000, 350000, 400000, 450000, 500000, 600000]  # House Prices in USD
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['House Size (sqft)', 'Bedrooms']]  # Input features: House Size and Bedrooms
y = df['Price']  # Target variable: House Price

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model

joblib.dump(model, r"D:\MiniProject\Trained Models\house_price_prediction_model.pkl")
# Print model coefficients and intercept
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Predict on the test set
y_pred = model.predict(X_test)
print("\nPredictions on Test Data:")
for i in range(len(X_test)):
    print(f"Input: House Size: {X_test.iloc[i]['House Size (sqft)']} sqft, Bedrooms: {X_test.iloc[i]['Bedrooms']} -> Predicted Price: ${y_pred[i]:,.2f}, Actual Price: ${y_test.iloc[i]:,.2f}")
