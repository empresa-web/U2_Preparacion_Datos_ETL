
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

datos = pd.read_csv(RESULTADOS / "datos_escalados.csv")
originales = pd.read_csv(
    RESULTADOS / "datos_no_supervisados.csv"
)

variables = [
    "cantidad_escalado",
    "precio_unitario_escalado",
    "total_venta_escalado"
]

X = datos[variables].to_numpy()


def evaluar(eps, min_samples):
    modelo = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    etiquetas = modelo.fit_predict(X)

    mascara = etiquetas != -1
    grupos = np.unique(etiquetas[mascara])
    n_clusters = len(grupos)
    ruido = int(np.sum(~mascara))
    cobertura = float(np.mean(mascara))

    resultado = {
        "eps": eps,
        "min_samples": min_samples,
        "Clusters": n_clusters,
        "Ruido": ruido,
        "Cobertura": cobertura,
        "Silhouette": np.nan,
        "Calinski_Harabasz": np.nan,
        "Davies_Bouldin": np.nan
    }

    # Evaluar solo grupos válidos, excluyendo ruido
    n_validos = int(mascara.sum())

    if 2 <= n_clusters < n_validos:
        X_validos = X[mascara]
        etiquetas_validas = etiquetas[mascara]

        resultado["Silhouette"] = silhouette_score(
            X_validos, etiquetas_validas
        )
        resultado["Calinski_Harabasz"] = (
            calinski_harabasz_score(
                X_validos, etiquetas_validas
            )
        )
        resultado["Davies_Bouldin"] = (
            davies_bouldin_score(
                X_validos, etiquetas_validas
            )
        )

    return modelo, etiquetas, resultado


# Modelo inicial
print("\n===== DBSCAN INICIAL =====")

modelo_inicial, etiquetas_iniciales, inicial = evaluar(
    eps=0.5,
    min_samples=5
)

print(pd.DataFrame([inicial]).round(4).to_string(index=False))

joblib.dump(
    modelo_inicial,
    MODELOS / "dbscan_inicial.joblib"
)

# Optimización de parámetros
print("\n===== OPTIMIZACION DBSCAN =====")

resultados = []

for eps in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]:
    for min_samples in [3, 5, 8]:

        _, _, resultado = evaluar(eps, min_samples)
        resultados.append(resultado)

metricas = pd.DataFrame(resultados)

metricas.to_csv(
    RESULTADOS / "optimizacion_dbscan.csv",
    index=False
)

print(metricas.round(4).to_string(index=False))

# Evitar elegir resultados con demasiado ruido
candidatos = metricas[
    (metricas["Clusters"] >= 2) &
    (metricas["Cobertura"] >= 0.80) &
    (metricas["Silhouette"].notna())
].copy()

if candidatos.empty:
    candidatos = metricas[
        (metricas["Clusters"] >= 2) &
        (metricas["Silhouette"].notna())
    ].copy()

if candidatos.empty:
    print("\nNo se encontro una configuracion con dos grupos validos.")
    raise SystemExit(1)

mejor = candidatos.sort_values(
    ["Silhouette", "Cobertura"],
    ascending=[False, False]
).iloc[0]

modelo_final, etiquetas_finales, final = evaluar(
    float(mejor["eps"]),
    int(mejor["min_samples"])
)

print("\n===== MEJOR CONFIGURACION DBSCAN =====")
print(pd.DataFrame([final]).round(4).to_string(index=False))

originales["cluster_dbscan"] = etiquetas_finales

originales.to_csv(
    RESULTADOS / "clusters_dbscan.csv",
    index=False
)

joblib.dump(
    modelo_final,
    MODELOS / "dbscan_optimizado.joblib"
)

print("\n===== DISTRIBUCION DE GRUPOS =====")
print(originales["cluster_dbscan"].value_counts().sort_index())

print("\nNota: el grupo -1 representa ruido o posibles valores atipicos.")

# Comparación inicial contra optimizado
comparacion = pd.DataFrame([
    {"Modelo": "DBSCAN inicial", **inicial},
    {"Modelo": "DBSCAN optimizado", **final}
])

comparacion.to_csv(
    RESULTADOS / "comparacion_dbscan.csv",
    index=False
)

# Visualización inicial y optimizada
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, etiquetas, titulo in [
    (axes[0], etiquetas_iniciales, "DBSCAN inicial"),
    (axes[1], etiquetas_finales, "DBSCAN optimizado")
]:
    mascara_ruido = etiquetas == -1

    ax.scatter(
        originales.loc[~mascara_ruido, "precio_unitario"],
        originales.loc[~mascara_ruido, "total_venta"],
        c=etiquetas[~mascara_ruido],
        cmap="viridis",
        s=55,
        alpha=0.8
    )

    if mascara_ruido.any():
        ax.scatter(
            originales.loc[mascara_ruido, "precio_unitario"],
            originales.loc[mascara_ruido, "total_venta"],
            c="red",
            marker="x",
            s=80,
            label="Ruido (-1)"
        )
        ax.legend()

    ax.set_title(titulo)
    ax.set_xlabel("Precio unitario (MXN)")
    ax.set_ylabel("Total de venta (MXN)")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    RESULTADOS / "comparacion_dbscan.png",
    dpi=300
)
plt.close()

print("\nModelos, metricas y grafica guardados correctamente.")
