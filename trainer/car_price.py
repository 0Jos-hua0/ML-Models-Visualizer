import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset: Horsepower, Weight vs. Car Price
data = {
    "Horsepower": [130, 150, 170, 200, 250, 300, 350, 400, 450, 500],
    "Weight": [3000, 3500, 3800, 4000, 4200, 4500, 4700, 4900, 5200, 5500],  # Weight in lbs
    "Price": [12000, 15000, 18000, 22000, 28000, 35000, 45000, 55000, 65000, 70000]  # Price in dollars
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Splitting features (Horsepower, Weight) and target variable (Price)
X = df[["Horsepower", "Weight"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, r"D:\MiniProject\models\CARmodel.pkl")

# Test the model
test_data = np.array([[250, 4000]])  # Example: Predict price for a car with 250 horsepower and 4000 lbs weight
predicted_price = model.predict(test_data)

print(f"Predicted car price for 250 horsepower and 4000 lbs weight: ${predicted_price[0]:,.2f}")
