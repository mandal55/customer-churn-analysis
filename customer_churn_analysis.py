"""
Customer Churn Analysis
Tech Stack: Python, Pandas, Scikit-learn

Loads customer data, trains a classification model to predict
whether a customer will churn (leave) or not.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------
data = pd.read_csv("data/customers.csv")
print("First 5 rows of dataset:")
print(data.head())

print("\nChurn counts:")
print(data["churn"].value_counts())

# -------------------------------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# -------------------------------------------------
le_contract = LabelEncoder()
data["contract_type_encoded"] = le_contract.fit_transform(data["contract_type"])

le_churn = LabelEncoder()
data["churn_encoded"] = le_churn.fit_transform(data["churn"])  # Yes=1, No=0

# -------------------------------------------------
# 3. FEATURES AND TARGET
# -------------------------------------------------
X = data[["age", "tenure_months", "monthly_charges", "total_charges",
          "contract_type_encoded", "support_calls"]]
y = data["churn_encoded"]

# -------------------------------------------------
# 4. TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------------------------
# 5. TRAIN MODEL
# -------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------
# 6. EVALUATE MODEL
# -------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_churn.classes_))

# -------------------------------------------------
# 7. FEATURE IMPORTANCE
# -------------------------------------------------
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")

# -------------------------------------------------
# 8. PREDICT FOR A NEW CUSTOMER (example)
# -------------------------------------------------
new_customer = pd.DataFrame({
    "age": [30],
    "tenure_months": [2],
    "monthly_charges": [85],
    "total_charges": [170],
    "contract_type_encoded": [le_contract.transform(["Month-to-Month"])[0]],
    "support_calls": [5]
})

prediction = model.predict(new_customer)
result = le_churn.inverse_transform(prediction)
print(f"\nPrediction for new customer: Churn = {result[0]}")
