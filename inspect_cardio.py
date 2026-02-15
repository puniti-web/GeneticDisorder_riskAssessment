import pandas as pd

# Load dataset
df = pd.read_csv("cardio.csv", sep=";")

print("\nFirst 5 rows:\n")
print(df.head())

print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nInfo:\n")
print(df.info())

print("\nTarget distribution:\n")
print(df["cardio"].value_counts())
