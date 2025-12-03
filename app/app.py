from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from extract_features import extract_features

app = Flask(__name__)

# Load models
models = {
    "Logistic Regression": pickle.load(open("Logistic_Regression.pkl", "rb")),
    "KNN": pickle.load(open("KNN.pkl", "rb")),
    "Decision Tree": pickle.load(open("Decision_Tree.pkl", "rb")),
    "Random Forest": pickle.load(open("Random_Forest.pkl", "rb")),
    "SVM": pickle.load(open("SVM.pkl", "rb"))
}
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]
    if not url.startswith(("http://", "https://")):
      url = "http://" + url
    model_name = request.form["model"]
    features = extract_features(url)
    print("Extracted features:", features)
    # Warn if too many features failed to extract
    if features.count(-1) > 25:
         warning = "⚠️ Warning: Many features could not be extracted. Please check the URL format or try a different link."
    else:
         warning = None
   # ✅ Load feature names used during training
    with open("feature_names.pkl", "rb") as f:
      feature_names = pickle.load(f)
      print("✅ Loaded feature names:", len(feature_names), feature_names)
      # Drop 'Port' if it was removed during training
      if "Port" in feature_names:
        port_index = feature_names.index("Port")
        del features[port_index]
        feature_names.remove("Port")
      print("✅ Extracted features:", len(features), features)
    if len(features) != len(feature_names):
      return render_template("index.html", prediction_text="❌ Feature mismatch. Please check extraction logic.", warning="Feature count mismatch.")
    df_input = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(df_input)
    print("Scaled features:", features_scaled)
    prediction = models[model_name].predict(features_scaled)[0]
    proba = models[model_name].predict_proba(features_scaled)[0]
    print("Confidence (Legitimate):", proba[1])
    print("Confidence (Phishing):", proba[0])
    print("Model used:", model_name)
    print("Prediction:", prediction)
    # result = "Phishing Website 🚨" if prediction == 1 else "Legitimate Website ✅"
    label = "Phishing Website 🚨" if prediction == -1 else "Legitimate Website ✅"
    confidence = round(proba[0] * 100, 2) if prediction == -1 else round(proba[1] * 100, 2)

    # return render_template("index.html", prediction_text=result)
    return render_template("index.html", prediction_text=label, warning=warning, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)