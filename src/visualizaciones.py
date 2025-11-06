import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("data/clean/icfes_clean_reducido.csv")
print(" Datos cargados:", df.shape)

os.makedirs("reports", exist_ok=True)

# --- Verificar columnas disponibles ---
print("Columnas disponibles:", list(df.columns)[:20])


# Distribución de puntajes por módulo

cols_puntajes = [c for c in df.columns if c.endswith("_PUNT")]
for col in cols_puntajes:
    plt.figure(figsize=(8,5))
    sns.histplot(df[col], bins=30, kde=True, color="skyblue")
    plt.title(f"Distribución del puntaje en {col}")
    plt.xlabel("Puntaje")
    plt.ylabel("Frecuencia")
    plt.savefig(f"reports/dist_{col}.png", dpi=120)
    plt.close()


corr = df[cols_puntajes].corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre módulos del Saber Pro")
plt.savefig("reports/matriz_correlacion.png", dpi=120)
plt.close()


if "ESTU_GENERO" in df.columns:
    df_melt = df.melt(id_vars="ESTU_GENERO", value_vars=cols_puntajes, var_name="MODULO", value_name="PUNTAJE")
    plt.figure(figsize=(10,5))
    sns.barplot(data=df_melt, x="MODULO", y="PUNTAJE", hue="ESTU_GENERO", palette="pastel", errorbar=None)
    plt.title("Promedio de puntajes por género y módulo")
    plt.xticks(rotation=45)
    plt.savefig("reports/puntaje_genero_modulo.png", dpi=120)
    plt.close()


if "FAMI_ESTRATOVIVIENDA" in df.columns:
    df_melt2 = df.melt(id_vars="FAMI_ESTRATOVIVIENDA", value_vars=cols_puntajes, var_name="MODULO", value_name="PUNTAJE")
    plt.figure(figsize=(10,5))
    sns.barplot(data=df_melt2, x="MODULO", y="PUNTAJE", hue="FAMI_ESTRATOVIVIENDA", palette="viridis", errorbar=None)
    plt.title("Promedio de puntajes por estrato y módulo")
    plt.xticks(rotation=45)
    plt.savefig("reports/puntaje_estrato_modulo.png", dpi=120)
    plt.close()

print(" Gráficos creados en la carpeta 'reports/'")
