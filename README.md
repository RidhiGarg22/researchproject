# 🧠 AI-Based Skin Health Predictor

An end-to-end machine learning project that predicts **Skin Health Score (SHS)** and **Dry Skin Level (DSL)** using lifestyle and environmental factors. The system includes trained regression models and a simple web interface for user input.

---

## 🚀 Project Overview

Skin health is influenced by daily habits such as sleep, hydration, stress, and skincare routine. This project applies machine learning regression techniques to:

* Predict **Skin Health Score (SHS)** (continuous value)
* Predict **Dry Skin Level (DSL)** (treated as numerical output)
* Analyze the impact of lifestyle factors on skin condition

---

## 🧩 Features

* 📊 Data preprocessing and feature scaling
* 🤖 Multiple regression models:

  * Random Forest Regressor
  * XGBoost Regressor
  * Support Vector Regressor (SVR)
* 📈 Model evaluation using:

  * R² Score
  * Mean Absolute Error (MAE)
* 📉 Data visualizations:

  * Correlation heatmap
  * Feature importance
  * Model comparison
  * Prediction graphs
* 🌐 Simple web interface for user input
* 💾 Model saving using joblib

---

## 🌐 Web Interface

A basic frontend allows users to input lifestyle parameters and get predictions.

### 🧾 Input Parameters

* Sleep Hours
* Sleep Quality (1–10)
* Stress Level (1–10)
* Water Intake (liters/day)
* Diet Type (1 = Healthy, 0 = Unhealthy)
* Screen Time (hours/day)
* Exercise (minutes/day)
* Skincare Routine (1 = Yes, 0 = No)
* Alcohol/Smoking (1 = Yes, 0 = No)

### ⚙️ Workflow

1. User enters input values
2. Data is scaled using a trained scaler
3. ML model predicts SHS and DSL
4. Results are displayed

---

## 🛠️ Tech Stack

* **Python**
* **pandas, numpy** → Data processing
* **scikit-learn** → ML models & preprocessing
* **XGBoost** → Boosting algorithm
* **matplotlib** → Visualization
* **joblib** → Model saving/loading
* **HTML** → Frontend

---

## 📂 Project Structure

```id="o4b7v0"
researchproject/
│
├── frontend/
│   └── index.html
│
├── train_model.py
├── backend.py
├── visualize_results.py
├── create_dataset.py
│
├── skin_health_dataset.csv
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── correlation_heatmap.png
│   ├── feature_importance_rf.png
│   ├── model_comparison_r2.png
│   ├── actual_vs_predicted_shs.png
│   ├── actual_vs_predicted_dsl.png
│   ├── shs_prediction_graph.png
│
└── .gitignore
```

---

## 📊 Model Performance

### 🔹 Random Forest

* SHS MAE: 3.06
* SHS R²: 0.9156
* DSL MAE: 0.0063
* DSL R²: 0.9971

---

### 🔹 XGBoost

* SHS MAE: 1.99
* SHS R²: 0.9650
* DSL MAE: 0.0120
* DSL R²: 0.9865

---

### 🔹 Support Vector Machine (SVM)

* SHS MAE: 2.30
* SHS R²: 0.9522
* DSL MAE: 0.2735
* DSL R²: 0.8053

---

### 🏆 Best Model

* **XGBoost** performs best for Skin Health Score prediction
* **Random Forest** performs best for Dry Skin Level prediction

---

## 📊 Outputs

The `outputs/` folder contains visualizations used for analysis:

* Correlation Heatmap
* Feature Importance (Random Forest)
* Model Comparison (R² Score)
* Actual vs Predicted Graphs

---


## 🎯 Future Improvements

* Convert to full web application using Flask or Streamlit
* Add real-time prediction API
* Improve dataset size and diversity
* Apply deep learning techniques

---

## 📌 Conclusion

This project demonstrates how regression-based machine learning models can effectively predict skin health using lifestyle data. The integration of a web interface makes the system interactive and practical for real-world applications.

---
