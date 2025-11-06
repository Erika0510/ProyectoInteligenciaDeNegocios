import pandas as pd

df = pd.read_csv("data/clean/icfes_clean.csv")

print("Filas:", len(df))
print("Columnas:", len(df.columns))
print("\n--- Descripción numérica ---\n")
print(df.describe())
print("\n--- Valores nulos por columna ---\n")
print(df.isna().sum().sort_values(ascending=False).head(15))
