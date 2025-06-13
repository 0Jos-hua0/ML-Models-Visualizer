import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Set the directory for trained models
MODEL_DIR = r"D:\MiniProject\Trained Models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Sample dataset (Replace with real data)
data = {
    "Age": [25, 40, 35, 23, 45, 55, 30, 60, 29, 50],
    "Caste": ["General", "OBC", "SC", "ST", "General", "SC", "OBC", "General", "ST", "SC"],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "Income": [30000, 60000, 50000, 25000, 75000, 90000, 45000, 100000, 32000, 85000],
    "Nationality": ["Indian", "Indian", "Foreign", "Indian", "Indian", "Foreign", "Indian", "Foreign", "Indian", "Indian"],
    "Employment Type": ["Salaried", "Self-Employed", "Salaried", "Self-Employed", "Salaried", "Self-Employed", "Salaried", "Self-Employed", "Salaried", "Self-Employed"],
    "Credit Score": ["High", "Medium", "Low", "Low", "High", "High", "Medium", "High", "Low", "Medium"],
    "Existing Loans": [1, 0, 2, 3, 0, 1, 2, 0, 3, 1],
    "Loan_Approval": [1, 1, 0, 0, 1, 1, 0, 1, 0, 1]  # 1 = Approved, 0 = Not Approved
}

df = pd.DataFrame(data)

# Encoding categorical variables (inside the code)
encoders = {}
for col in ["Caste", "Gender", "Nationality", "Employment Type", "Credit Score"]:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Splitting features and target variable
X = df.drop(columns=["Loan_Approval"])
y = df["Loan_Approval"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = os.path.join(MODEL_DIR, "loan_decision_tree.pkl")
joblib.dump(model, model_path)

print(f"Model training complete. Saved to {model_path}\n")


# Function to predict loan approval + max loan amount
def predict_loan(input_data):
    """
    Predicts loan approval and maximum loan amount for a given applicant.

    :param input_data: dict containing Age, Caste, Gender, Income, Nationality, Employment Type, Credit Score, Existing Loans
    :return: dict with Loan Status and Maximum Loan Amount
    """
    # Ensure required keys are present
    required_keys = ["Age", "Caste", "Gender", "Income", "Nationality", "Employment Type", "Credit Score", "Existing Loans"]
    if not all(key in input_data for key in required_keys):
        return {"error": "Missing required fields"}

    # Encoding categorical values
    encoded_input = [
        input_data["Age"],
        input_data["Income"],
        input_data["Existing Loans"]
    ]
    for feature in ["Caste", "Gender", "Nationality", "Employment Type", "Credit Score"]:
        if input_data[feature] in encoders[feature].classes_:
            encoded_input.append(encoders[feature].transform([input_data[feature]])[0])
        else:
            return {"error": f"Invalid value for {feature}: {input_data[feature]}"}

    # Convert to NumPy array and reshape for prediction
    input_array = np.array(encoded_input).reshape(1, -1)

    # Load the trained model
    model = joblib.load(model_path)

    # Predict loan approval
    prediction = model.predict(input_array)[0]
    result = "Approved" if prediction == 1 else "Not Approved"

    # Calculate maximum loan amount based on income and credit score
    loan_multipliers = {"Low": 2, "Medium": 5, "High": 8}
    max_loan_amount = input_data["Income"] * loan_multipliers.get(input_data["Credit Score"], 2)

    return {
        "Loan Status": result,
        "Maximum Loan Amount": f"₹{max_loan_amount:,}" if prediction == 1 else "Not Available"
    }


# Example usage
sample_input = {
    "Age": 30,
    "Caste": "OBC",
    "Gender": "Male",
    "Income": 65000,
    "Nationality": "Indian",
    "Employment Type": "Salaried",
    "Credit Score": "Medium",
    "Existing Loans": 1
}

output = predict_loan(sample_input)
print(output)
