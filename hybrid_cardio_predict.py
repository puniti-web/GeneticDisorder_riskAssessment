import pickle
import numpy as np
import pandas as pd

# Load pipeline
pipeline = pickle.load(open("cardio_pipeline.pkl", "rb"))

# -------------------------
# Example Patient Input
# -------------------------

age = 55
gender = 1          # 1=female, 2=male
ap_hi = 140         # systolic BP
ap_lo = 90          # diastolic BP
cholesterol = 2     # 1=normal, 2=above normal, 3=well above normal
gluc = 1            # 1=normal, 2=above normal, 3=well above normal
smoke = 1           # 0=no, 1=yes
alco = 0            # 0=no, 1=yes
active = 0          # 0=no, 1=yes
bmi = 28

# Create dataframe in correct order
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

# Get base ML probability
base_prob = pipeline.predict_proba(patient_df)[0][1]
# Get base ML probability
base_prob = pipeline.predict_proba(patient_df)[0][1]

print("\nBase ML Cardiovascular Risk Probability:", round(base_prob, 3))

# -------------------------
# Feature Importance (Explainability)
# -------------------------

model = pipeline.named_steps["model"]
feature_names = pipeline.named_steps["scaler"].feature_names_in_

importances = model.feature_importances_

importance_df = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop Contributing Risk Factors:")
for feature, score in importance_df[:5]:
    print(f"- {feature} (importance score: {round(score,3)})")



# -------------------------
# Genetic Risk Modifier
# -------------------------

relatives = int(input("\nNumber of first-degree relatives with premature CVD (0,1,2+): "))

if relatives == 0:
    RR = 1.0
elif relatives == 1:
    RR = 1.6
else:
    RR = 2.3

# Epidemiology-inspired adjustment
adjusted_risk = 1 - (1 - base_prob) ** RR

print("\nRelative Risk Multiplier Applied:", RR)
print("Adjusted Genetic Risk Probability:", round(adjusted_risk, 3))

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

# Cap maximum probability
adjusted_risk = min(adjusted_risk, 0.95)

print("\n=========================================")
print(" HYBRID GENETIC CARDIOVASCULAR RISK REPORT")
print("=========================================")

print(f"\nBase ML Risk Estimate: {round(base_prob*100,2)}%")

if relatives > 0:
    print(f"Genetic Risk Multiplier Applied: {RR}x")
    print(f"Adjusted Genetic Risk: {round(adjusted_risk*100,2)}%")
else:
    print("No genetic amplification applied.")

print(f"\nFinal Risk Category: {level}")

print("\nInterpretation:")
if level == "Low Risk":
    print("- Maintain healthy lifestyle.")
elif level == "Borderline Risk":
    print("- Monitor blood pressure and cholesterol regularly.")
elif level == "Elevated Risk":
    print("- Lifestyle modification strongly recommended.")
else:
    print("- Clinical evaluation and medical consultation advised.")

print("\nDisclaimer:")
print("This tool provides probabilistic risk screening.")
print("It is NOT a medical diagnosis.")
print("=========================================\n")

