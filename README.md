# 🛡️ Phishing Website Detection using ML Algorithms

A machine learning project to detect phishing websites using multiple classifiers (Logistic Regression, KNN, Decision Tree, Random Forest, SVM).  
The system is trained on extracted features from URLs and HTML content, then saved as `.pkl` models for fast inference.

---

## 📂 Project Structure

phishing-website-detection/
│
├── phishing_data.csv              # Raw dataset
├── phishing_data_cleaned.csv      # Cleaned dataset (auto-generated)
├── train_models.py                # Training script
├── predict.py                     # Inference script
├── extract_features.py            # Feature extraction from raw URLs
│
├── Logistic_Regression.pkl        # Trained Logistic Regression model
├── KNN.pkl                        # Trained KNN model
├── Decision_Tree.pkl              # Trained Decision Tree model
├── Random_Forest.pkl              # Trained Random Forest model
├── SVM.pkl                        # Trained SVM model
│
├── scaler.pkl                     # Saved StandardScaler for preprocessing
├── feature_names.pkl              # Saved feature schema for inference
└── README.md                      # Project documentation
---

## ⚙️ Setup

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd phishing-website-detection


2. Install dependencies:
pip install -r requirements.txt
Dependencies include:
- pandas
- numpy
- scikit-learn
- requests
- beautifulsoup4
- python-whois


3. Prepare dataset:
Place phishing_data.csv in the project root.
## 🚀 Training Models
Run the training script:

        ```bash
        python train_models.py

This will:
- Clean and preprocess the dataset
- Engineer the Missing_Ratio feature
- Train 5 classifiers
- Print train vs. test accuracy and classification reports
- Save trained models (.pkl), scaler, and feature names for inference
Artifacts created:
- Logistic_Regression.pkl
- KNN.pkl
- Decision_Tree.pkl
- Random_Forest.pkl
- SVM.pkl
- scaler.pkl
- feature_names.pkl

## 🔎 Making Predictions:
        Run the training script:

        ```bash
        python train_models.py

This will:
- Load a trained model (e.g., Random Forest)
- Load the scaler and feature schema
- Take a sample row from the cleaned dataset
- Predict whether it’s phishing (-1) or legitimate (1)

You can also adapt predict.py to use extract_features(url) for real-time URL checks.

## 📊 Models Used:
- Logistic Regression → interpretable linear baseline
- KNN → distance-based classifier (requires scaling)
- Decision Tree → interpretable non-linear model
- Random Forest → ensemble of trees, reduces overfitting
- SVM (RBF kernel) → powerful non-linear classifier

## 🧠 Key Features:
- Automatic column cleaning (spaces, hyphens, typos fixed)
- Missing_Ratio feature to capture incomplete data
- Scaling with StandardScaler for sensitive models
- Feature importance reports for interpretability
- Reusable .pkl models for fast inference

## 📌 Notes:
- -1 → Phishing website
- 1 → Legitimate website
- Models are saved in .pkl format for portability
- Use extract_features.py to generate features from raw URLs during deployment

## 👨‍💻 Author
**Developed by:** MHB Shimla  
**Location:** Sylhet, Bangladesh  
**Focus:** Robust, modular, and professional ML systems for phishing detection.

