
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

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

# Cargar datasets
datos = pd.read_csv(RESULTADOS / "datos_escalados.csv")
originales = pd.read_csv(
    RESULTADOS / "datos_no_supervisados.csv"
)

variables = [
    "cantidad_escalado",
    "precio_unitario_escalado",
    "total_venta_escalado"
]

X = datos[variables]

resultados = []

print("\n===== OPTIMIZACION DE K-MEANS =====")

# Probar diferentes cantidades de clusters
for k in range(2, 9):

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    etiquetas = modelo.fit_predict(X)

    resultados.append({
        "Clusters": k,
        "Silhouette": silhouette_score(X, etiquetas),
        "Calinski_Harabasz": calinski_harabasz_score(
            X, etiquetas
        ),
        "Davies_Bouldin": davies_bouldin_score(
            X, etiquetas
        ),
        "Inercia": modelo.inertia_
    })

# Comparar resultados
metricas = pd.DataFrame(resultados)

print(metricas.round(4).to_string(index=False))

metricas.to_csv(
    RESULTADOS / "optimizacion_kmeans.csv",
    index=False
)

# Elegir el mejor numero de clusters
mejor = metricas.loc[metricas["Silhouette"].idxmax()]
mejor_k = int(mejor["Clusters"])

print("\n===== MEJOR CONFIGURACION =====")
print("Numero de clusters:", mejor_k)
print("Silhouette Score:", round(mejor["Silhouette"], 4))
print("Davies-Bouldin:", round(mejor["Davies_Bouldin"], 4))
print("Calinski-Harabasz:", round(mejor["Calinski_Harabasz"], 4))

# Entrenar modelo optimizado
modelo_final = KMeans(
    n_clusters=mejor_k,
    random_state=42,
    n_init=20
)

etiquetas_finales = modelo_final.fit_predict(X)

originales["cluster"] = etiquetas_finales

originales.to_csv(
    RESULTADOS / "clusters_kmeans_optimizado.csv",
    index=False
)

joblib.dump(
    modelo_final,
    MODELOS / "kmeans_optimizado.joblib"
)

print("\n===== DISTRIBUCION DE CLUSTERS =====")
print(originales["cluster"].value_counts().sort_index())

print("\n===== PROMEDIOS POR CLUSTER =====")
print(
    originales.groupby("cluster")[
        ["cantidad", "precio_unitario", "total_venta"]
    ].mean().round(2)
)

# Grafica de optimizacion
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(
    metricas["Clusters"],
    metricas["Silhouette"],
    marker="o"
)
axes[0].set_title("Silhouette Score")
axes[0].set_xlabel("Numero de clusters")
axes[0].grid(alpha=0.3)

axes[1].plot(
    metricas["Clusters"],
    metricas["Davies_Bouldin"],
    marker="o",
    color="orange"
)
axes[1].set_title("Davies-Bouldin (menor es mejor)")
axes[1].set_xlabel("Numero de clusters")
axes[1].grid(alpha=0.3)

axes[2].plot(
    metricas["Clusters"],
    metricas["Inercia"],
    marker="o",
    color="green"
)
axes[2].set_title("Metodo del codo")
axes[2].set_xlabel("Numero de clusters")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    RESULTADOS / "optimizacion_kmeans.png",
    dpi=300
)
plt.close()

# Visualizar clusters optimizados
plt.figure(figsize=(10, 6))

grafica = plt.scatter(
    originales["precio_unitario"],
    originales["total_venta"],
    c=etiquetas_finales,
    cmap="viridis",
    s=65,
    alpha=0.8
)

plt.title(f"K-Means optimizado: {mejor_k} clusters")
plt.xlabel("Precio unitario (MXN)")
plt.ylabel("Total de venta (MXN)")
plt.colorbar(grafica, label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTADOS / "kmeans_optimizado.png",
    dpi=300
)
plt.close()

print("\nOptimizacion, modelo y graficas guardados correctamente.")
