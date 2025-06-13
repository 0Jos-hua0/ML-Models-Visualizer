import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample dataset (Predicting house prices based on area)
data = {
    "Area": [750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400],
    "Price": [150000, 160000, 170000, 180000, 190000, 200000, 220000, 240000, 260000, 280000]
}

df = pd.DataFrame(data)

# Splitting features and target variable
X = df[["Area"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, r"D:\MiniProject\Trained Models\linear_regression.pkl")

print("Model trained and saved successfully!")

# Example Prediction
area_input = np.array([[1050]])  # Predict for 1050 sq.ft
predicted_price = model.predict(area_input)
print(f"Predicted Price for {area_input[0][0]} sq.ft: ₹{predicted_price[0]:,.2f}")
