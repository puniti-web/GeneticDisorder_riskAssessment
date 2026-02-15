# 🫀 Hybrid AI Genetic Disease (Cardiovascular) Risk Assessment System

A full-stack machine learning web application that estimates cardiovascular disease risk using a hybrid approach combining:

* Clinical machine learning prediction
* Genetic risk amplification logic
* AI-powered personalized preventive guidance
* Explainable feature importance
* Interactive health-tech dashboard UI

---

## 🚀 Project Overview

This system predicts cardiovascular disease probability using structured clinical features and enhances the prediction using a family-history risk multiplier.

It is designed as a **screening tool**, not a diagnostic system.

The project demonstrates:

* End-to-end ML pipeline development
* Feature engineering
* Model evaluation with cross-validation
* Explainability integration
* Hybrid risk modeling (ML + domain logic)
* AI-assisted medical explanation
* Full-stack deployment with Flask
* Secure API key handling
* Modern SaaS-style dashboard UI

---

## 🧠 Architecture

### 1️⃣ Machine Learning Layer

* Dataset: 70,000 patient cardiovascular records

* Features:

  * Age
  * Gender
  * BMI
  * Systolic BP (ap_hi)
  * Diastolic BP (ap_lo)
  * Cholesterol level
  * Blood glucose
  * Smoking
  * Alcohol
  * Physical activity

* Model: Tree-based classifier (via sklearn Pipeline)

* Scaling: StandardScaler

* Evaluation:

  * 5-fold cross-validation
  * Mean CV Accuracy: ~70.9%
  * Test Accuracy: ~71%

---

### 2️⃣ Hybrid Genetic Risk Amplification

We apply a relative risk multiplier based on family history:

| Family History            | Multiplier |
| ------------------------- | ---------- |
| None                      | 1.0x       |
| One first-degree relative | 1.6x       |
| Two or more               | 2.3x       |

Explanation:

Why 1.0x, 1.6x, and 2.3x?

These values are grounded in epidemiological evidence:

🔹 1.0x — No Family History

If no first-degree relatives (parent/sibling) have premature cardiovascular disease, no amplification is applied.

RR = 1.0


This preserves the original ML probability.

🔹 1.6x — One First-Degree Relative

Multiple cardiovascular studies report that having one first-degree relative with premature cardiovascular disease increases risk by approximately 1.5–2.0 times.

We selected:

RR = 1.6


This represents moderate but clinically meaningful amplification without causing probability inflation.

🔹 2.3x — Two or More First-Degree Relatives

Risk increases significantly when multiple first-degree relatives are affected.

Literature suggests:

2.0–3.0x increased risk depending on age of onset and clustering.

We selected:

RR = 2.3


This reflects substantial inherited risk while keeping model output numerically stable.

Why Use This Formula?

We apply:

Adjusted Risk = 1 - (1 - Base Probability)^RR


Instead of simply multiplying probability.

Why?

Because direct multiplication:

0.70 × 2.3 = 1.61  (invalid probability)


Using survival-based amplification:

Keeps values between 0 and 1

Preserves probabilistic interpretation

Prevents unrealistic inflation

Aligns with risk compounding models used in epidemiology

Adjusted risk formula:

```
Adjusted Risk = 1 - (1 - Base Probability)^RR
```

This preserves probabilistic interpretation while increasing genetic influence.

---

### 3️⃣ Explainability Layer

Global feature importance is extracted from the trained model.

Top contributors are displayed in the UI to increase transparency.

Example output:

* Age
* BMI
* Systolic BP
* Diastolic BP
* Cholesterol

---

### 4️⃣ AI-Powered Preventive Guidance

We integrate Google Gemini API to generate:

* Risk explanation
* Lifestyle recommendations
* Medical monitoring advice
* Preventive precautions

AI output is rendered using Markdown for professional formatting.

API keys are securely stored in `.env` and excluded via `.gitignore`.

---

## 💻 Tech Stack

Backend:

* Python
* Flask
* scikit-learn
* Pandas
* NumPy

Frontend:

* Bootstrap 5
* Chart.js
* Modern SaaS dashboard styling

AI:

* Google Gemini 2.5 Flash

Security:

* dotenv for environment variables
* .gitignore for key protection

---

## 📊 Dashboard Features

* Interactive risk donut visualization
* Risk category classification
* Personalized AI-generated medical guidance
* Genetic multiplier transparency
* Clean health-tech startup UI

---

## ⚠️ Disclaimer

This system is a probabilistic screening tool.

It is not intended for clinical diagnosis or treatment decisions.

Users are advised to consult qualified healthcare professionals.

---

## 📂 Project Structure

```
├── app.py
├── ai_explainer.py
├── cardio_pipeline.pkl
├── templates/
│   └── index.html
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔒 Security

* `.env` file is excluded from version control.
* API keys are not stored in the repository.
* Model file can be excluded if needed for deployment.

---

## 🧪 How To Run Locally

```
git clone <repo-url>
cd project-folder
pip install -r requirements.txt
```

Create `.env`:

```
GEMINI_API_KEY=your_key_here
```

Run:

```
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 📈 Future Improvements

* SHAP-based per-patient explainability
* Model calibration
* Input validation with medical ranges
* PDF report generation
* User authentication
* Persistent patient records


#Made by Puniti Jodhwani
