
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

# Cargar datos preparados
datos = pd.read_csv(
    RESULTADOS / "datos_escalados.csv"
)

originales = pd.read_csv(
    RESULTADOS / "datos_no_supervisados.csv"
)

variables = [
    "cantidad_escalado",
    "precio_unitario_escalado",
    "total_venta_escalado"
]

X = datos[variables]

# Entrenar modelo inicial con 3 clusters
modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

etiquetas = modelo.fit_predict(X)

# Evaluar agrupamiento
silhouette = silhouette_score(X, etiquetas)
calinski = calinski_harabasz_score(X, etiquetas)
davies = davies_bouldin_score(X, etiquetas)

# Guardar los grupos asignados
originales["cluster"] = etiquetas

originales.to_csv(
    RESULTADOS / "clusters_kmeans_inicial.csv",
    index=False
)

# Guardar modelo entrenado
joblib.dump(
    modelo,
    MODELOS / "kmeans_inicial.joblib"
)

# Mostrar resultados
print("\n===== K-MEANS INICIAL =====")
print("Numero de clusters:", modelo.n_clusters)
print("Registros analizados:", len(X))

print("\n===== METRICAS =====")
print(f"Silhouette Score: {silhouette:.4f}")
print(f"Calinski-Harabasz: {calinski:.4f}")
print(f"Davies-Bouldin: {davies:.4f}")

print("\n===== DISTRIBUCION DE CLUSTERS =====")
print(originales["cluster"].value_counts().sort_index())

print("\n===== PROMEDIOS POR CLUSTER =====")
print(
    originales.groupby("cluster")[
        ["cantidad", "precio_unitario", "total_venta"]
    ].mean().round(2)
)

# Guardar metricas
metricas = pd.DataFrame([{
    "Modelo": "KMeans inicial",
    "Clusters": 3,
    "Silhouette": silhouette,
    "Calinski_Harabasz": calinski,
    "Davies_Bouldin": davies
}])

metricas.to_csv(
    RESULTADOS / "metricas_kmeans_inicial.csv",
    index=False
)

# Grafica inicial
plt.figure(figsize=(10, 6))

grafica = plt.scatter(
    originales["precio_unitario"],
    originales["total_venta"],
    c=etiquetas,
    cmap="viridis",
    s=65,
    alpha=0.8
)

plt.title("K-Means inicial: agrupacion de ventas")
plt.xlabel("Precio unitario (MXN)")
plt.ylabel("Total de venta (MXN)")
plt.colorbar(grafica, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTADOS / "kmeans_inicial.png",
    dpi=300
)

plt.close()

print("\nModelo, metricas y grafica guardados correctamente.")
