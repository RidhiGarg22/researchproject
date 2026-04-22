import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

# Load dataset
df = pd.read_csv("skin_health_dataset.csv")

# Load trained models and scaler
skin_score_model = joblib.load("skin_health_score_model.pkl")
dry_skin_model = joblib.load("dry_skin_model.pkl")
features = joblib.load("feature_names.pkl")
scaler = joblib.load("scaler.pkl")

# Prepare X and y
X = df[features]
y_score = df["skin_health_score"]
y_dry = df["dry_skin_level"]

# Scale X
X_scaled = scaler.transform(X)

# Feature importance for skin health score
importance_score = skin_score_model.feature_importances_
sorted_idx = np.argsort(importance_score)
plt.figure(figsize=(8,6))
plt.barh(np.array(features)[sorted_idx], importance_score[sorted_idx], color='skyblue')
plt.title("Feature Importance - Skin Health Score")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Feature importance for dry skin
importance_dry = dry_skin_model.feature_importances_
sorted_idx_dry = np.argsort(importance_dry)
plt.figure(figsize=(8,6))
plt.barh(np.array(features)[sorted_idx_dry], importance_dry[sorted_idx_dry], color='salmon')
plt.title("Feature Importance - Dry Skin Risk")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# SHAP for skin score
explainer_score = shap.TreeExplainer(skin_score_model)
shap_values_score = explainer_score.shap_values(X_scaled)
shap.summary_plot(shap_values_score, X_scaled, feature_names=features, plot_type="bar")

# SHAP for dry skin
explainer_dry = shap.TreeExplainer(dry_skin_model)
shap_values_dry = explainer_dry.shap_values(X_scaled)
shap.summary_plot(shap_values_dry[2], X_scaled, feature_names=features, plot_type="bar")
