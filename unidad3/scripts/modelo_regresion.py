from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad3" / "resultados"
MODELOS = ROOT / "unidad3" / "modelos"

# Cargar datos
train = pd.read_csv(RESULTADOS / "entrenamiento.csv")
test = pd.read_csv(RESULTADOS / "prueba.csv")

# Variables predictoras
categoricas = ["producto", "sucursal"]
numericas = ["precio_unitario", "mes", "dia_semana"]
variables = categoricas + numericas

X_train = train[variables]
X_test = test[variables]

y_train = train["total_venta"]
y_test = test["total_venta"]

# Preprocesamiento
preprocesador = ColumnTransformer([
    ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
    ("numericas", StandardScaler(), numericas)
])

# Modelos
algoritmos = {
    "Base mediana": DummyRegressor(strategy="median"),
    "Regresion Ridge": Ridge(alpha=1.0)
}

metricas = []

for nombre, algoritmo in algoritmos.items():

    modelo = Pipeline([
        ("preprocesador", preprocesador),
        ("regresor", algoritmo)
    ])

    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    metricas.append({
        "Modelo": nombre,
        "MAE": mean_absolute_error(y_test, predicciones),
        "RMSE": root_mean_squared_error(y_test, predicciones),
        "R2": r2_score(y_test, predicciones)
    })

    if nombre == "Regresion Ridge":

        joblib.dump(modelo, MODELOS / "regresion_ridge.joblib")

        plt.figure(figsize=(10, 5))
        plt.plot(y_test.values, label="Ventas reales", marker="o")
        plt.plot(predicciones, label="Ventas predichas", marker="s")
        plt.title("Regresion Ridge: ventas reales y predichas")
        plt.xlabel("Registro de prueba")
        plt.ylabel("Importe en MXN")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS / "regresion_ridge.png", dpi=300)
        plt.close()

# Guardar resultados
reporte = pd.DataFrame(metricas)
reporte.to_csv(RESULTADOS / "metricas_regresion.csv", index=False)

print("\n===== RESULTADOS DE REGRESION =====")
print(reporte.to_string(index=False))
print("\nModelo y grafica guardados correctamente.")