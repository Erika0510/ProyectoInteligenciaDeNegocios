import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from joblib import load

# Cargar datos y modelo
df = pd.read_csv("data/clean/icfes_clean_reducido.csv")
clf = load("data/warehouse/modelo_rendimiento.joblib")


PUNTS = [c for c in [
    "MOD_LECTURA_CRITICA_PUNT",
    "MOD_RAZONA_CUANTITATIVO_PUNT",
    "MOD_COMUNI_ESCRITA_PUNT",
    "MOD_INGLES_PUNT",
    "MOD_COMPETEN_CIUDADA_PUNT",
] if c in df.columns]

CATS = [c for c in ["ESTU_GENERO","FAMI_ESTRATOVIVIENDA","ESTU_DEPTO_RESIDE","ESTU_HORASSEMANATRABAJA"] if c in df.columns]

if "HORAS_TRABAJO_NUM" in df.columns and "ESTU_HORASSEMANATRABAJA" not in CATS:
    
    extra_num = ["HORAS_TRABAJO_NUM"]
else:
    extra_num = []


expected_cols = list(set(PUNTS + CATS + extra_num))
for col in expected_cols:
    if col not in df.columns:
        df[col] = pd.NA 

X = df[expected_cols].copy()

y = pd.qcut(df[PUNTS].mean(axis=1), q=3, labels=["Bajo","Medio","Alto"])


idx = X.sample(12000, random_state=7).index
pred = clf.predict(X.loc[idx])

# Matriz de confusión
cm = confusion_matrix(y.loc[idx], pred, labels=["Bajo","Medio","Alto"])
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bajo","Medio","Alto"],
            yticklabels=["Bajo","Medio","Alto"])
plt.title("Matriz de confusión")
plt.ylabel("Real"); plt.xlabel("Predicho")
plt.tight_layout()
plt.savefig("reports/matriz_confusion_modelo.png", dpi=140)
print("Figura guardada en reports/matriz_confusion_modelo.png")
