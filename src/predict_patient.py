import pandas as pd
import numpy as np
import joblib
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# ==== Load Models and Encoders ====

# Load trained model
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load full dummy feature list BEFORE RFECV
with open("full_feature_columns.pkl", "rb") as f:
    full_columns = pickle.load(f)

# Load RFECV selected feature mask
with open("selected_features_mask.pkl", "rb") as f:
    selected_mask = pickle.load(f)

# ==== Gather User Input ====

print("Enter patient info below:")
age = int(input("Age: "))
gender = input("Gender (M/F): ").lower()
race = input("Race (white/black/asian/other): ").lower()
ethnicity = input("Ethnicity (hispanic/non-hispanic): ").lower()
smoking_status = input(
    "Smoking Status (never smoker/current smoker/former smoker): ").lower()
pregnant = input("Pregnant? (yes/no): ").lower()
presenting_problem = input(
    "Presenting problem (e.g., chest pain, headache): ").lower()

# ==== Build Feature Row ====

user_data = pd.DataFrame([{
    'age': age,
    'GENDER': gender,
    'RACE': race,
    'ETHNICITY': ethnicity,
    'SMOKING_STATUS': smoking_status,
    'PREGNANT': pregnant,
    'PRESENTING_PROBLEM': presenting_problem
}])

# Dummy encode user input
user_encoded = pd.get_dummies(user_data)

# Make sure all dummy columns are present
for col in full_columns:
    if col not in user_encoded:
        user_encoded[col] = 0

# Reorder to match training full feature order
user_encoded = user_encoded[full_columns]

# ==== Scale Full Dummy Feature Set ====

user_scaled_full = scaler.transform(user_encoded)

# ==== Apply RFECV Feature Selection AFTER Scaling ====

user_scaled = user_scaled_full[:, selected_mask]

# ==== Predict ====

predicted_class = model.predict(user_scaled)[0]
predicted_med = label_encoder.inverse_transform([predicted_class])[0]

print(f"\n Predicted Medication: {predicted_med}")

# (Optional) Show prediction confidence
proba = model.predict_proba(user_scaled)[0]
confidence = np.max(proba) * 100
print(f"Confidence: {confidence:.2f}%")
