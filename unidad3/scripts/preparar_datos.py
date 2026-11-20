from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad3" / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

# Cargar datos limpios de la Unidad 2
df = pd.read_csv(ROOT / "datos" / "ventas_limpias.csv")

# Generar variables temporales
df["fecha"] = pd.to_datetime(df["fecha"])
df["mes"] = df["fecha"].dt.month
df["dia_semana"] = df["fecha"].dt.dayofweek

# Variable objetivo de clasificación
df["demanda_alta"] = (df["cantidad"] >= 3).astype(int)

# Seleccionar variables para el análisis
columnas = [
    "id_venta",
    "producto",
    "sucursal",
    "precio_unitario",
    "mes",
    "dia_semana",
    "total_venta",
    "demanda_alta"
]

datos = df[columnas].copy()

# Dividir los datos
entrenamiento, prueba = train_test_split(
    datos,
    test_size=0.20,
    random_state=42,
    stratify=datos["demanda_alta"]
)

# Guardar archivos
datos.to_csv(RESULTADOS / "datos_supervisados.csv", index=False)
entrenamiento.to_csv(RESULTADOS / "entrenamiento.csv", index=False)
prueba.to_csv(RESULTADOS / "prueba.csv", index=False)

# Mostrar resultados
print("\n===== PREPARACION DE DATOS =====")
print(f"Registros totales: {len(datos)}")
print(f"Entrenamiento: {len(entrenamiento)}")
print(f"Prueba: {len(prueba)}")

print("\n===== DISTRIBUCION DE DEMANDA =====")
print(datos["demanda_alta"].value_counts())

print("\n===== VISTA PREVIA =====")
print(datos.head().to_string(index=False))

print("\nArchivos generados correctamente.")