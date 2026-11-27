from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

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

y_train = train["demanda_alta"]
y_test = test["demanda_alta"]

# Crear Pipeline
def crear_modelo(algoritmo):

    preprocesador = ColumnTransformer([
        ("categoricas",
         OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("numericas", StandardScaler(), numericas)
    ])

    return Pipeline([
        ("preprocesador", preprocesador),
        ("clasificador", algoritmo)
    ])

# Modelos iniciales
algoritmos = {
    "Base mayoritaria": DummyClassifier(
        strategy="most_frequent"
    ),
    "Regresion logistica": LogisticRegression(
        max_iter=2000
    ),
    "Random Forest inicial": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

modelos = {}

for nombre, algoritmo in algoritmos.items():

    print(f"Entrenando {nombre}...")

    modelo = crear_modelo(algoritmo)
    modelo.fit(X_train, y_train)

    modelos[nombre] = modelo

# Optimizacion
parametros = {
    "clasificador__n_estimators": [100, 200],
    "clasificador__max_depth": [2, 4, None],
    "clasificador__min_samples_leaf": [2, 5, 10]
}

print("\nOptimizando Random Forest...")

busqueda = GridSearchCV(
    crear_modelo(RandomForestClassifier(random_state=42)),
    parametros,
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    scoring="f1",
    n_jobs=2
)

busqueda.fit(X_train, y_train)

modelos["Random Forest optimizado"] = busqueda.best_estimator_

print("\n===== MEJORES HIPERPARAMETROS =====")
print(busqueda.best_params_)

# Evaluacion
metricas = []

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for (nombre, modelo), ax in zip(
    modelos.items(), axes.flatten()
):

    predicciones = modelo.predict(X_test)

    metricas.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, predicciones),
        "Precision": precision_score(
            y_test, predicciones, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predicciones, zero_division=0
        ),
        "F1": f1_score(
            y_test, predicciones, zero_division=0
        )
    })

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicciones,
        labels=[0, 1],
        display_labels=["Baja", "Alta"],
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    ax.set_title(nombre)

    archivo = nombre.lower().replace(" ", "_")

    joblib.dump(modelo, MODELOS / f"{archivo}.joblib")

plt.tight_layout()
plt.savefig(
    RESULTADOS / "matrices_confusion.png",
    dpi=300
)
plt.close()

# Guardar metricas
reporte = pd.DataFrame(metricas)

reporte.to_csv(
    RESULTADOS / "metricas_clasificacion.csv",
    index=False
)

print("\n===== RESULTADOS DE CLASIFICACION =====")
print(reporte.to_string(index=False))

print("\n===== REPORTE RANDOM FOREST OPTIMIZADO =====")

pred_opt = modelos["Random Forest optimizado"].predict(X_test)

print(classification_report(
    y_test,
    pred_opt,
    labels=[0, 1],
    target_names=["Demanda baja", "Demanda alta"],
    zero_division=0
))

print("\nModelos, metricas y matrices guardados correctamente.")