
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

# Cargar metricas
kmeans_inicial = pd.read_csv(
    RESULTADOS / "metricas_kmeans_inicial.csv"
)

kmeans_opt = pd.read_csv(
    RESULTADOS / "optimizacion_kmeans.csv"
)

dbscan = pd.read_csv(
    RESULTADOS / "comparacion_dbscan.csv"
)

pca = joblib.load(MODELOS / "modelo_pca.joblib")

# Seleccionar el K-Means optimizado
mejor_kmeans = kmeans_opt.loc[
    kmeans_opt["Silhouette"].idxmax()
]

# Consolidar comparacion
comparacion = pd.DataFrame([
    {
        "Modelo": "K-Means inicial",
        "Clusters": int(kmeans_inicial.iloc[0]["Clusters"]),
        "Ruido": 0,
        "Cobertura": 1.0,
        "Silhouette": kmeans_inicial.iloc[0]["Silhouette"],
        "Calinski_Harabasz": kmeans_inicial.iloc[0]["Calinski_Harabasz"],
        "Davies_Bouldin": kmeans_inicial.iloc[0]["Davies_Bouldin"]
    },
    {
        "Modelo": "K-Means optimizado",
        "Clusters": int(mejor_kmeans["Clusters"]),
        "Ruido": 0,
        "Cobertura": 1.0,
        "Silhouette": mejor_kmeans["Silhouette"],
        "Calinski_Harabasz": mejor_kmeans["Calinski_Harabasz"],
        "Davies_Bouldin": mejor_kmeans["Davies_Bouldin"]
    }
])

columnas = comparacion.columns.tolist()
comparacion = pd.concat(
    [comparacion, dbscan[columnas]],
    ignore_index=True
)

comparacion.to_csv(
    RESULTADOS / "comparacion_final_modelos.csv",
    index=False
)

print("\n===== COMPARACION FINAL =====")
print(comparacion.round(4).to_string(index=False))

# Resumen PCA
varianza = pca.explained_variance_ratio_

print("\n===== RESULTADOS PCA =====")
print(f"Componente 1: {varianza[0]*100:.2f}%")
print(f"Componente 2: {varianza[1]*100:.2f}%")
print(f"Varianza acumulada: {varianza.sum()*100:.2f}%")

# Grafica comparativa de metricas
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metricas = [
    ("Silhouette", "Mayor es mejor"),
    ("Calinski_Harabasz", "Mayor es mejor"),
    ("Davies_Bouldin", "Menor es mejor")
]

for ax, (columna, criterio) in zip(axes, metricas):
    ax.bar(
        comparacion["Modelo"],
        comparacion[columna]
    )
    ax.set_title(f"{columna}\n{criterio}")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    RESULTADOS / "comparacion_metricas_final.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Grafica de grupos y cobertura
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(
    comparacion["Modelo"],
    comparacion["Clusters"]
)
axes[0].set_title("Cantidad de clusters")
axes[0].set_ylabel("Numero de grupos")
axes[0].tick_params(axis="x", rotation=25)

axes[1].bar(
    comparacion["Modelo"],
    comparacion["Cobertura"] * 100
)
axes[1].set_title("Cobertura de registros")
axes[1].set_ylabel("Porcentaje (%)")
axes[1].set_ylim(0, 110)
axes[1].tick_params(axis="x", rotation=25)

plt.tight_layout()

plt.savefig(
    RESULTADOS / "comparacion_grupos_cobertura.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\n===== INTERPRETACION =====")
print("K-Means asigna todos los registros a un cluster.")
print("DBSCAN puede excluir registros considerados ruido.")
print("Las metricas de DBSCAN excluyen el ruido.")
print("Por ello, las puntuaciones no son directamente equivalentes.")
print("PCA se utiliza para reducir dimensiones y visualizar grupos.")

print("\nArchivos comparativos generados correctamente.")
