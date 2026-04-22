from flask_cors import CORS
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# -----------------------------
# 1️⃣ Create Flask app
# -----------------------------
app = Flask(__name__)
CORS(app)
# -----------------------------
# 2️⃣ Load trained models & scaler
# -----------------------------
skin_model = joblib.load("skin_health_score_model.pkl")
dry_model = joblib.load("dry_skin_model.pkl")
features = joblib.load("feature_names.pkl")   # column names
scaler = joblib.load("scaler.pkl")           # scaler used during training

# -----------------------------
# 3️⃣ Define API route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Receive JSON input from frontend
        data = request.json

        # 2. Convert to DataFrame in correct order
        df = pd.DataFrame([data])[features]

        # 3. Scale the input
        df_scaled = scaler.transform(df)

        # 4. Predict skin health score
        skin_score = skin_model.predict(df_scaled)[0]

        # 5. Predict dry skin level
        dry_level = dry_model.predict(df_scaled)[0]
        dry_labels = ["No Dryness", "Mild Dryness", "Severe Dryness"]

        # 6. Return as JSON
        return jsonify({
            "skin_score": round(skin_score, 2),
            "dry_skin": dry_labels[dry_level]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------------
# 4️⃣ Run the app
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5050)

