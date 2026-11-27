
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[2]
DATOS = ROOT / "datos"
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

RESULTADOS.mkdir(parents=True, exist_ok=True)
MODELOS.mkdir(parents=True, exist_ok=True)

# Cargar datos limpios de la Unidad 2
df = pd.read_csv(DATOS / "ventas_limpias.csv")

print("\n===== DATOS ORIGINALES DEL ANALISIS =====")
print("Registros:", len(df))
print("Columnas:", len(df.columns))

# Seleccionar características numéricas
variables = [
    "cantidad",
    "precio_unitario",
    "total_venta"
]

datos = df[["id_venta", "producto", "sucursal"] + variables].copy()

# Comprobar valores faltantes
print("\n===== VALORES FALTANTES =====")
print(datos.isnull().sum())

# Eliminar registros incompletos si existieran
datos = datos.dropna(subset=variables)

# Guardar dataset seleccionado
datos.to_csv(
    RESULTADOS / "datos_no_supervisados.csv",
    index=False
)

# Estandarización de características
escalador = StandardScaler()
datos_escalados = escalador.fit_transform(datos[variables])

# Crear dataset transformado
transformados = pd.DataFrame(
    datos_escalados,
    columns=[f"{col}_escalado" for col in variables]
)

transformados.insert(0, "id_venta", datos["id_venta"].values)

transformados.to_csv(
    RESULTADOS / "datos_escalados.csv",
    index=False
)

# Guardar escalador
joblib.dump(
    escalador,
    MODELOS / "escalador.joblib"
)

print("\n===== PREPARACION COMPLETADA =====")
print("Registros procesados:", len(datos))
print("Variables utilizadas:", variables)

print("\n===== DATOS ESTANDARIZADOS =====")
print(transformados.head().to_string(index=False))

print("\nArchivos generados correctamente.")
