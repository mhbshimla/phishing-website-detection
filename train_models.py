import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle


# Load raw dataset
df = pd.read_csv("phishing_data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

# Fix known typos
df.rename(columns={
    "URLURL_Length": "URL_Length",
    "having_IPhaving_IP_Address": "having_IP_Address"
}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(-1, inplace=True)

# Optional: Save cleaned version
df.to_csv("phishing_data_cleaned.csv", index=False)
print("✅ Data cleaned and saved as phishing_data_cleaned.csv")

# Drop non-feature columns
if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

# Rename columns to match feature extraction
df.rename(columns={
    "having_IP_Address": "Having_IP_Address",
    "Shortining_Service": "Shortening_Service",
    "having_At_Symbol": "At_Symbol",
    "double_slash_redirecting": "Redirecting",
    "Prefix_Suffix": "Prefix_Suffix",
    "having_Sub_Domain": "Subdomain_Count",
    "SSLfinal_State": "SSL_Final_State",
    "Domain_registeration_length": "Domain_Registration_Length",
    "port": "Port",
    "HTTPS_token": "HTTPS_Token",
    "Links_in_tags": "Links_in_Tags",
    "Submitting_to_email": "Submitting_to_Email",
    "on_mouseover": "On_Mouseover",
    "popUpWidnow": "PopUp_Window",
    "age_of_domain": "Domain_Age",
    "DNSRecord": "DNS_Resolution",
    "web_traffic": "Web_Traffic",
    "Links_pointing_to_page": "Links_Pointing_to_Page",
    "Statistical_report": "Statistical_Report"
}, inplace=True)


# Check for missing values
if df.isnull().sum().sum() > 0:
    print("⚠️ Missing values detected. Filling with mode.")
    df.fillna(df.mode().iloc[0], inplace=True)
# print("📊 Columns in dataset:", df.columns.tolist())

# Separate features and target
X = df.drop("Result", axis=1)
y = df["Result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Replace -1 with 0 to match inference-time preprocessing
X_train = X_train.replace(-1, 0)
X_test = X_test.replace(-1, 0)

# Simulate missing features in training data
import random

def simulate_missing_features(X, drop_count=20):
    X_sim = X.copy()
    for i in range(len(X_sim)):
        drop_indices = random.sample(range(X_sim.shape[1]), drop_count)
        for idx in drop_indices:
            X_sim.iloc[i, idx] = 0
    return X_sim

# Use clean data (no dropout)
X_train_sim = X_train.copy()
X_test_sim = X_test.copy()

# Add Missing_Ratio BEFORE scaling
X_train_sim["Missing_Ratio"] = X_train_sim.apply(lambda row: (row == 0).sum() / len(row), axis=1)
X_test_sim["Missing_Ratio"] = X_test_sim.apply(lambda row: (row == 0).sum() / len(row), axis=1)

# Drop one feature to keep total at 30
X_train_sim.drop("Port", axis=1, inplace=True)
X_test_sim.drop("Port", axis=1, inplace=True)

# Save feature names for inference
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X_train_sim.columns.tolist(), f)


# Scale after both have same columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sim)
X_test_scaled = scaler.transform(X_test_sim)

# Save the fitted scaler for inference
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Unscaled sample:\n", X_train_sim.iloc[0])
print("Scaled sample:\n", X_train_scaled[0])

# Define models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=8),
    # "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Decision_Tree": DecisionTreeClassifier(max_depth=50, min_samples_leaf=1, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True)
}

# Train, evaluate, and save each model

scaled_models = ["Logistic_Regression", "KNN", "SVM"]

for name, model in models.items():
    print(f"\n🔍 Training {name}")
    
    # Train on scaled or unscaled data depending on model
    if name in scaled_models:
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train_sim, y_train)
        y_train_pred = model.predict(X_train_sim)
        y_test_pred = model.predict(X_test_sim)

    # Save trained model back into dict
    models[name] = model  

    # Print train vs test accuracy
    print("📊 Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("📊 Test Accuracy:", accuracy_score(y_test, y_test_pred))

    # Print classification report for test set
    print(classification_report(y_test, y_test_pred))

    # Feature importance
    feature_names = X_train_sim.columns
    if name == "Logistic_Regression":
        coefs = np.abs(model.coef_[0])
        top_features = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]
        print("\n🌟 Top 10 Features by Weight (Logistic Regression):")
        for fname, score in top_features:
            print(f"{fname}: {score:.4f}")

    if name in ["Random_Forest", "Decision_Tree"]:
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🌟 Top 10 Features by Importance ({name}):")
        for fname, score in top_features:
            print(f"{fname}: {score:.4f}")

    # Save model
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

