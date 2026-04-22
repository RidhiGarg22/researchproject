# ================= STEP 0: Import Libraries =================
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


# ================= STEP 1: Load Dataset =================
df = pd.read_csv("skin_health_dataset.csv")


# ================= Correlation Heatmap =================
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap of Lifestyle Factors and Skin Health")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()


# ================= STEP 2: Features & Targets =================
X = df.drop(columns=["skin_health_score", "dry_skin_level"])

y_score = df["skin_health_score"]     # SHS
y_dsl = df["dry_skin_level"]          # DSL


# ================= STEP 3: Train-Test Split =================
X_train, X_test, y_score_train, y_score_test, y_dsl_train, y_dsl_test = train_test_split(
    X, y_score, y_dsl, test_size=0.2, random_state=42
)


# ================= STEP 4: Feature Scaling =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ================= STEP 5: Initialize Models =================
rf_reg_score = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg_dsl = RandomForestRegressor(n_estimators=200, random_state=42)

xgb_reg_score = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb_reg_dsl = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

svr_score = SVR(kernel="rbf", C=100, gamma="scale")
svr_dsl = SVR(kernel="rbf", C=100, gamma="scale")


# ================= STEP 6: Train Models =================
rf_reg_score.fit(X_train_scaled, y_score_train)
rf_reg_dsl.fit(X_train_scaled, y_dsl_train)

xgb_reg_score.fit(X_train_scaled, y_score_train)
xgb_reg_dsl.fit(X_train_scaled, y_dsl_train)

svr_score.fit(X_train_scaled, y_score_train)
svr_dsl.fit(X_train_scaled, y_dsl_train)


# ================= STEP 7: Evaluation =================
print("\n========== RANDOM FOREST ==========")
print("SHS MAE:", mean_absolute_error(y_score_test, rf_reg_score.predict(X_test_scaled)))
print("SHS R2:", r2_score(y_score_test, rf_reg_score.predict(X_test_scaled)))

print("DSL MAE:", mean_absolute_error(y_dsl_test, rf_reg_dsl.predict(X_test_scaled)))
print("DSL R2:", r2_score(y_dsl_test, rf_reg_dsl.predict(X_test_scaled)))


print("\n========== XGBOOST ==========")
print("SHS MAE:", mean_absolute_error(y_score_test, xgb_reg_score.predict(X_test_scaled)))
print("SHS R2:", r2_score(y_score_test, xgb_reg_score.predict(X_test_scaled)))

print("DSL MAE:", mean_absolute_error(y_dsl_test, xgb_reg_dsl.predict(X_test_scaled)))
print("DSL R2:", r2_score(y_dsl_test, xgb_reg_dsl.predict(X_test_scaled)))


print("\n========== SVM ==========")
print("SHS MAE:", mean_absolute_error(y_score_test, svr_score.predict(X_test_scaled)))
print("SHS R2:", r2_score(y_score_test, svr_score.predict(X_test_scaled)))

print("DSL MAE:", mean_absolute_error(y_dsl_test, svr_dsl.predict(X_test_scaled)))
print("DSL R2:", r2_score(y_dsl_test, svr_dsl.predict(X_test_scaled)))


# ================= STEP 7.5: Feature Importance (Random Forest) =================
feature_importance = rf_reg_score.feature_importances_
features = X.columns

indices = np.argsort(feature_importance)

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance for Skin Health Score")

plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=300)
plt.show()


# ================= STEP 7.6: Model Comparison Bar Chart =================
rf_r2 = r2_score(y_score_test, rf_reg_score.predict(X_test_scaled))
xgb_r2 = r2_score(y_score_test, xgb_reg_score.predict(X_test_scaled))
svm_r2 = r2_score(y_score_test, svr_score.predict(X_test_scaled))

models = ["Random Forest", "XGBoost", "SVM"]
scores = [rf_r2, xgb_r2, svm_r2]

plt.figure(figsize=(7,5))
plt.bar(models, scores)

plt.ylabel("R² Score")
plt.title("Model Comparison for Skin Health Score Prediction")

for i, v in enumerate(scores):
    plt.text(i, v + 0.01, round(v, 3), ha="center")

plt.tight_layout()
plt.savefig("model_comparison_r2.png", dpi=300)
plt.show()


# ================= STEP 7.7: Actual vs Predicted (Skin Health Score) =================
y_pred_shs = xgb_reg_score.predict(X_test_scaled)

plt.figure(figsize=(6,6))
plt.scatter(y_score_test, y_pred_shs, alpha=0.7)

plt.plot(
    [y_score_test.min(), y_score_test.max()],
    [y_score_test.min(), y_score_test.max()],
    'r--'
)

plt.xlabel("Actual Skin Health Score")
plt.ylabel("Predicted Skin Health Score")
plt.title("Actual vs Predicted Skin Health Score")

plt.tight_layout()
plt.savefig("actual_vs_predicted_shs.png", dpi=300)
plt.show()


# ================= STEP 7.8: Actual vs Predicted (Dry Skin Level) =================
y_pred_dsl = xgb_reg_dsl.predict(X_test_scaled)

plt.figure(figsize=(6,6))
plt.scatter(y_dsl_test, y_pred_dsl, alpha=0.7)

plt.plot(
    [y_dsl_test.min(), y_dsl_test.max()],
    [y_dsl_test.min(), y_dsl_test.max()],
    'r--'
)

plt.xlabel("Actual Dry Skin Level")
plt.ylabel("Predicted Dry Skin Level")
plt.title("Actual vs Predicted Dry Skin Level")

plt.tight_layout()
plt.savefig("actual_vs_predicted_dsl.png", dpi=300)
plt.show()


# ================= STEP 8: Save Models =================
joblib.dump(rf_reg_score, "rf_skin_health_score.pkl")
joblib.dump(rf_reg_dsl, "rf_dry_skin_level.pkl")

joblib.dump(xgb_reg_score, "xgb_skin_health_score.pkl")
joblib.dump(xgb_reg_dsl, "xgb_dry_skin_level.pkl")

joblib.dump(svr_score, "svm_skin_health_score.pkl")
joblib.dump(svr_dsl, "svm_dry_skin_level.pkl")

joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\n✅ Regression models for SHS and DSL trained and saved successfully!")