import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Create sample data
np.random.seed(42)
n = 200
parents_salary = np.random.randint(10000, 200000, size=n)
first_graduate = np.random.choice([0, 1], size=n)

# Simulate target: 0 = Low fees, 1 = High fees
college_fees = []
for sal, fg in zip(parents_salary, first_graduate):
    if sal > 100000 and fg == 0:
        college_fees.append(1)
    elif sal > 150000:
        college_fees.append(1)
    else:
        college_fees.append(0)

# 2. DataFrame
df = pd.DataFrame({
    'parents_salary': parents_salary,
    'first_graduate': first_graduate,
    'college_fees_high': college_fees
})

# 3. Split features and target
X = df[['parents_salary', 'first_graduate']]
y = df['college_fees_high']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 5. Create pipeline with scaler + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# 6. Train pipeline
pipeline.fit(X_train, y_train)

# 7. Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# 8. Save pipeline
joblib.dump(pipeline, r'D:\MiniProject\models\logistic_pipeline(boolean).pkl')


# 9. Optional: Test with sample input
sample = pd.DataFrame({
    'parents_salary': [120000, 50000],
    'first_graduate': [1, 0]
})
print("Sample predictions:", pipeline.predict(sample))
