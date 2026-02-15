from flask import Flask, render_template, request
import pickle
import pandas as pd
from ai_explainer import generate_precautions
import markdown


app = Flask(__name__)

# Load trained model
pipeline = pickle.load(open("cardio_pipeline.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    gender = int(request.form["gender"])
    ap_hi = int(request.form["ap_hi"])
    ap_lo = int(request.form["ap_lo"])
    cholesterol = int(request.form["cholesterol"])
    gluc = int(request.form["gluc"])
    smoke = int(request.form["smoke"])
    alco = int(request.form["alco"])
    active = int(request.form["active"])
    bmi = float(request.form["bmi"])
    relatives = int(request.form["relatives"])

    patient_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "bmi": bmi
    }])

    # -------------------------
    # ML Prediction
    # -------------------------
    base_prob = pipeline.predict_proba(patient_df)[0][1]

    # -------------------------
    # Genetic Multiplier
    # -------------------------
    if relatives == 0:
        RR = 1.0
    elif relatives == 1:
        RR = 1.6
    else:
        RR = 2.3

    adjusted_risk = 1 - (1 - base_prob) ** RR
    adjusted_risk = min(adjusted_risk, 0.95)

    risk_percent = round(adjusted_risk * 100, 2)

    # -------------------------
    # Risk Category
    # -------------------------
    if adjusted_risk < 0.25:
        level = "Low Risk"
    elif adjusted_risk < 0.50:
        level = "Borderline Risk"
    elif adjusted_risk < 0.75:
        level = "Elevated Risk"
    else:
        level = "High Risk"

    # -------------------------
    # Feature Importance
    # -------------------------
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["scaler"].feature_names_in_
    importances = model.feature_importances_

    importance_df = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = ", ".join([f[0] for f in importance_df[:3]])
    genetic_flag = "Yes" if relatives > 0 else "No"

    # -------------------------
    # AI Explanation
    # -------------------------
    ai_text = generate_precautions(level, top_features, genetic_flag)
    ai_text = markdown.markdown(ai_text)



    return render_template(
        "index.html",
        prediction=risk_percent,
        ai_text=ai_text
    )

if __name__ == "__main__":
    app.run(debug=True)
