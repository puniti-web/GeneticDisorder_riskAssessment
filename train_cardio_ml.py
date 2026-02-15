import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
df = pd.read_csv("cardio.csv", sep=";")

# Drop useless column
df = df.drop("id", axis=1)

# Convert age from days to years
df["age"] = df["age"] / 365

# Create BMI (patient-friendly feature)
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

# Drop raw height and weight (since BMI is more interpretable)
df = df.drop(["height", "weight"], axis=1)

# Features and target
X = df.drop("cardio", axis=1)
y = df["cardio"]

# Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)

print("\nCross-Validation Accuracy Scores:", scores)
print("Mean CV Accuracy:", np.mean(scores))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save pipeline
pickle.dump(pipeline, open("cardio_pipeline.pkl", "wb"))

print("\nCardiovascular ML pipeline saved successfully.")
