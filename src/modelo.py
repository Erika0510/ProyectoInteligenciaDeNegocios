import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Config
DATA_PATH = "data/clean/icfes_clean_reducido.csv"
MODEL_PATH = "data/warehouse/modelo_rendimiento.joblib"
PRED_PATH  = "data/warehouse/predicciones_rendimiento.csv"

# Módulos de puntaje 
PUNTS = [
    "MOD_LECTURA_CRITICA_PUNT",
    "MOD_RAZONA_CUANTITATIVO_PUNT",
    "MOD_COMUNI_ESCRITA_PUNT",
    "MOD_INGLES_PUNT",
    "MOD_COMPETEN_CIUDADA_PUNT",
]

# Variables categóricas/numéricas adicionales
CATS = [c for c in ["ESTU_GENERO", "FAMI_ESTRATOVIVIENDA", "ESTU_DEPTO_RESIDE", "ESTU_HORASSEMANATRABAJA"]]
NUMS = []  


#Cargar datos
df = pd.read_csv(DATA_PATH)

PUNTS = [c for c in PUNTS if c in df.columns]
CATS  = [c for c in CATS  if c in df.columns]
NUMS  = [c for c in NUMS  if c in df.columns]

if len(PUNTS) < 3:
    raise ValueError(f"No hay suficientes columnas de puntaje. Encontradas: {PUNTS}")

#  variable objetivo (Bajo/Medio/Alto)
df["PUNT_PROM"] = df[PUNTS].mean(axis=1, skipna=True)

q33 = df["PUNT_PROM"].quantile(0.33)
q66 = df["PUNT_PROM"].quantile(0.66)

def label_rend(x):
    if x <= q33: return "Bajo"
    if x <= q66: return "Medio"
    return "Alto"

df["RENDIMIENTO"] = df["PUNT_PROM"].apply(label_rend)


X = df[PUNTS + CATS + NUMS].copy()
y = df["RENDIMIENTO"].copy()

# Preprocesamiento 
numeric_features   = PUNTS + NUMS
categorical_features = CATS

numeric_transformer = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh",  OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

#  Modelo 
clf = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

#Train / Test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print("\n== Métricas ==")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, pred, labels=["Bajo","Medio","Alto"]))
print("\nReporte de clasificación:\n", classification_report(y_test, pred, digits=3))

# Guardar modelo
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"\n Modelo guardado en: {MODEL_PATH}")

# recomendaciones
def recomendar_area(row):
   
    modulos = {m: row.get(m, np.nan) for m in PUNTS}
    mejor = max(modulos, key=lambda k: modulos[k] if pd.notna(modulos[k]) else -1)

    mapping = {
        "MOD_RAZONA_CUANTITATIVO_PUNT": "Ingenierías, Economía, Estadística",
        "MOD_LECTURA_CRITICA_PUNT": "Derecho, Ciencias Sociales, Humanidades",
        "MOD_COMUNI_ESCRITA_PUNT": "Comunicación Social, Lenguas, Educación",
        "MOD_INGLES_PUNT": "Idiomas, Relaciones Internacionales, Turismo",
        "MOD_COMPETEN_CIUDADA_PUNT": "Ciencia Política, Administración Pública, Educación Cívica",
    }
    return mapping.get(mejor, "Explorar intereses y refuerzo transversal")


sample = df.sample(n=min(5000, len(df)), random_state=42).copy()
X_s = sample[PUNTS + CATS + NUMS]
sample["PRED_RENDIMIENTO"] = clf.predict(X_s)
sample["RECOMENDACION_AREA"] = sample.apply(recomendar_area, axis=1)

cols_export = CATS + NUMS + PUNTS + ["PUNT_PROM", "RENDIMIENTO", "PRED_RENDIMIENTO", "RECOMENDACION_AREA"]
cols_export = [c for c in cols_export if c in sample.columns]

sample[cols_export].to_csv(PRED_PATH, index=False)
print(f"Predicciones + recomendaciones guardadas en: {PRED_PATH}")


rf = clf.named_steps["rf"]
feature_names = clf.named_steps["prep"].get_feature_names_out()
importancias = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
importancias.to_csv("data/warehouse/importancia_variables.csv", index=False)
print("Importancia guardada en data/warehouse/importancia_variables.csv")

