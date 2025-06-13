import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset: Age, Income (in ₹1000), Purchased (1 = Yes, 0 = No)
data = {
    "Age": [22, 25, 47, 52, 46, 56, 25, 23, 50, 43, 35, 48, 51, 40, 60],
    "Income": [30000, 40000, 70000, 80000, 65000, 90000, 42000, 38000, 75000, 60000, 50000, 72000, 81000, 55000, 95000],
    "Purchased": [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features (X) and target variable (y)
X = df[["Age", "Income"]]
y = df["Purchased"]

# Normalize features for better accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a KNN classifier
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save the trained model and scaler
joblib.dump(model, r"D:\MiniProject\Trained Models\model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModel saved as knn_product_purchase.pkl")

# Function to take user input and predict purchase behavior
def predict_purchase():
    print("\nEnter customer details:")
    age = float(input("Age: "))
    income = float(input("Monthly Income (in ₹): "))

    # Normalize input using the saved scaler
    scaler = joblib.load("scaler.pkl")
    input_data = scaler.transform([[age, income]])

    # Load the trained model
    model = joblib.load("knn_product_purchase.pkl")

    # Predict purchase decision
    prediction = model.predict(input_data)[0]
    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"

    print(f"\nPrediction: {result}")

# Call function to take user input and predict
predict_purchase()
