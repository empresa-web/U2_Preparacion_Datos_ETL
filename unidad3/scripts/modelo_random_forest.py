from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad3" / "resultados"
MODELOS = ROOT / "unidad3" / "modelos"

# Cargar datos
train = pd.read_csv(RESULTADOS / "entrenamiento.csv")
test = pd.read_csv(RESULTADOS / "prueba.csv")

categoricas = ["producto", "sucursal"]
numericas = ["precio_unitario", "mes", "dia_semana"]
variables = categoricas + numericas

X_train = train[variables]
X_test = test[variables]

y_train = train["total_venta"]
y_test = test["total_venta"]

# Preprocesamiento y modelo
def crear_modelo():
    preprocesador = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("numericas", StandardScaler(), numericas)
    ])

    return Pipeline([
        ("preprocesador", preprocesador),
        ("regresor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

# Entrenamiento inicial
print("Entrenando Random Forest inicial...")

modelo_inicial = crear_modelo()
modelo_inicial.fit(X_train, y_train)

pred_inicial = modelo_inicial.predict(X_test)

# Optimizacion de hiperparametros
parametros = {
    "regresor__n_estimators": [100, 200],
    "regresor__max_depth": [2, 4, None],
    "regresor__min_samples_leaf": [1, 3, 5]
}

print("Optimizando hiperparametros...")

busqueda = GridSearchCV(
    crear_modelo(),
    parametros,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_mean_absolute_error",
    n_jobs=2
)

busqueda.fit(X_train, y_train)

modelo_optimizado = busqueda.best_estimator_
pred_optimizado = modelo_optimizado.predict(X_test)

print("\n===== MEJORES HIPERPARAMETROS =====")
print(busqueda.best_params_)

# Evaluacion
def evaluar(nombre, predicciones):
    return {
        "Modelo": nombre,
        "MAE": mean_absolute_error(y_test, predicciones),
        "RMSE": root_mean_squared_error(y_test, predicciones),
        "R2": r2_score(y_test, predicciones)
    }

metricas = pd.DataFrame([
    evaluar("Random Forest inicial", pred_inicial),
    evaluar("Random Forest optimizado", pred_optimizado)
])

print("\n===== RESULTADOS RANDOM FOREST =====")
print(metricas.to_string(index=False))

# Guardar resultados y modelos
metricas.to_csv(
    RESULTADOS / "metricas_random_forest.csv",
    index=False
)

joblib.dump(
    modelo_inicial,
    MODELOS / "regresion_random_forest_inicial.joblib"
)

joblib.dump(
    modelo_optimizado,
    MODELOS / "regresion_random_forest_optimizado.joblib"
)

# Grafica
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Ventas reales", marker="o")
plt.plot(pred_optimizado, label="Ventas predichas", marker="s")
plt.title("Random Forest optimizado")
plt.xlabel("Registro de prueba")
plt.ylabel("Importe en MXN")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTADOS / "random_forest_optimizado.png",
    dpi=300
)

plt.close()

print("\nModelos, metricas y grafica guardados correctamente.")