import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = 3 * X[:, 0] + 2 * X[:, 1] + 5

# Introduce NaN values in some target labels to simulate a faulty model
y_corrupt = y.copy()
y_corrupt[::10] = np.nan  # Every 10th value is NaN

# Train a Linear Regression model
model = LinearRegression()
try:
    model.fit(X, y_corrupt)  # This will cause an error
except ValueError as e:
    print(f"Error during training: {e}")

# Force-save the model even if it's faulty

joblib.dump(model, r"D:\MiniProject\Trained Models\faulty_model.pkl")
print("Faulty model saved as 'faulty_model.pkl'.")
