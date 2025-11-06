import pandas as pd
import os

# Ruta del archivo fuente
file_path = "data/raw/Resultados_únicos_Saber_Pro_20251023.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encontró el archivo en {file_path}")

# Cargar el archivo
df = pd.read_csv(file_path, low_memory=False, nrows=50000)

print("Archivo cargado correctamente:", df.shape)
print("Columnas iniciales:", df.columns[:10])

# Normalizar nombres de columnas
df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

# Eliminar duplicados
df = df.drop_duplicates()

# Reemplazar valores nulos comunes
df = df.replace(["NA", "N/A", "", "SIN INFORMACION"], pd.NA)


# Guardar archivo limpio
output_path = "data/clean/icfes_clean.csv"
df.to_csv(output_path, index=False)

print("💾 Archivo limpio guardado en:", output_path)

# columnas clave
cols_interes = [
    "PERIODO", "ESTU_GENERO", "ESTU_DEPTO_RESIDE", "ESTU_MCPIO_RESIDE",
    "ESTU_HORASSEMANATRABAJA", "FAMI_ESTRATOVIVIENDA", "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEINTERNET", "FAMI_TIENEAUTOMOVIL", "FAMI_EDUCACIONPADRE",
    "FAMI_EDUCACIONMADRE", "ESTU_COLE_TERMINO", "PUNT_GLOBAL",
    "MOD_LECTURA_CRITICA_PUNT", "MOD_RAZONA_CUANTITATIVO_PUNT",
    "MOD_COMUNI_ESCRITA_PUNT", "MOD_INGLES_PUNT", "MOD_COMPETEN_CIUDADA_PUNT"
]

df_filtrado = df[[c for c in cols_interes if c in df.columns]]
df_filtrado.to_csv("data/clean/icfes_clean_reducido.csv", index=False)
print("Guardado dataset reducido con columnas clave.")

