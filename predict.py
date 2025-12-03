import pickle
import numpy as np
import pandas as pd

# Load artifacts
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Choose a model to test (e.g., Random Forest)
with open("Random_Forest.pkl", "rb") as f:
    model = pickle.load(f)

# Load cleaned dataset
df = pd.read_csv("phishing_data_cleaned.csv")

# Take one sample (row) for testing
sample = df.drop("Result", axis=1).iloc[0]   # first row features
sample = sample.replace(-1, 0)               # match preprocessing
sample["Missing_Ratio"] = (sample == 0).sum() / len(sample)

# Convert to DataFrame with correct columns
sample_df = pd.DataFrame([sample], columns=feature_names)

# Scale features
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)
print("Prediction:", prediction[0])