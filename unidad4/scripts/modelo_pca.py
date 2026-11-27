
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.decomposition import PCA

# Rutas del proyecto
ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad4" / "resultados"
MODELOS = ROOT / "unidad4" / "modelos"

# Cargar datasets
datos = pd.read_csv(
    RESULTADOS / "datos_escalados.csv"
)

kmeans = pd.read_csv(
    RESULTADOS / "clusters_kmeans_optimizado.csv"
)

dbscan = pd.read_csv(
    RESULTADOS / "clusters_dbscan.csv"
)

variables = [
    "cantidad_escalado",
    "precio_unitario_escalado",
    "total_venta_escalado"
]

X = datos[variables]

# Aplicar PCA con dos componentes
modelo_pca = PCA(n_components=2)

componentes = modelo_pca.fit_transform(X)

# Varianza explicada
varianza = modelo_pca.explained_variance_ratio_

print("\n===== ANALISIS PCA =====")
print("Registros analizados:", len(X))
print("Variables originales:", len(variables))
print("Componentes principales:", 2)

print("\n===== VARIANZA EXPLICADA =====")
print(f"Componente 1: {varianza[0]:.4f}")
print(f"Componente 2: {varianza[1]:.4f}")
print(f"Varianza acumulada: {sum(varianza):.4f}")

# Crear dataset transformado
resultado = pd.DataFrame({
    "id_venta": datos["id_venta"],
    "PC1": componentes[:, 0],
    "PC2": componentes[:, 1],
    "cluster_kmeans": kmeans["cluster"],
    "cluster_dbscan": dbscan["cluster_dbscan"]
})

resultado.to_csv(
    RESULTADOS / "datos_pca.csv",
    index=False
)

# Guardar modelo PCA
joblib.dump(
    modelo_pca,
    MODELOS / "modelo_pca.joblib"
)

# Tabla de componentes
pesos = pd.DataFrame(
    modelo_pca.components_,
    columns=variables,
    index=["PC1", "PC2"]
)

print("\n===== PESOS DE LOS COMPONENTES =====")
print(pesos.round(4).to_string())

pesos.to_csv(
    RESULTADOS / "componentes_pca.csv"
)

# Grafica de varianza explicada
plt.figure(figsize=(8, 5))

plt.bar(
    ["PC1", "PC2"],
    varianza * 100
)

plt.title("PCA: varianza explicada")
plt.ylabel("Porcentaje de varianza (%)")
plt.ylim(0, 100)

for i, valor in enumerate(varianza):
    plt.text(
        i,
        valor * 100 + 1,
        f"{valor * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    RESULTADOS / "varianza_pca.png",
    dpi=300
)
plt.close()

# Comparacion visual de agrupamientos
fig, axes = plt.subplots(
    1, 2, figsize=(14, 6)
)

# K-Means en dos dimensiones
grafica1 = axes[0].scatter(
    resultado["PC1"],
    resultado["PC2"],
    c=resultado["cluster_kmeans"],
    cmap="viridis",
    s=60,
    alpha=0.8
)

axes[0].set_title("K-Means optimizado con PCA")
axes[0].set_xlabel("Componente principal 1")
axes[0].set_ylabel("Componente principal 2")
axes[0].grid(alpha=0.3)

fig.colorbar(
    grafica1,
    ax=axes[0],
    label="Cluster"
)

# DBSCAN en dos dimensiones
mascara = resultado["cluster_dbscan"] != -1

grafica2 = axes[1].scatter(
    resultado.loc[mascara, "PC1"],
    resultado.loc[mascara, "PC2"],
    c=resultado.loc[mascara, "cluster_dbscan"],
    cmap="viridis",
    s=60,
    alpha=0.8
)

# Mostrar ruido
axes[1].scatter(
    resultado.loc[~mascara, "PC1"],
    resultado.loc[~mascara, "PC2"],
    color="red",
    marker="x",
    s=80,
    label="Ruido (-1)"
)

axes[1].set_title("DBSCAN optimizado con PCA")
axes[1].set_xlabel("Componente principal 1")
axes[1].set_ylabel("Componente principal 2")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.colorbar(
    grafica2,
    ax=axes[1],
    label="Cluster"
)

plt.tight_layout()
plt.savefig(
    RESULTADOS / "comparacion_pca.png",
    dpi=300
)
plt.close()

print("\n===== ARCHIVOS GENERADOS =====")
print("datos_pca.csv")
print("componentes_pca.csv")
print("varianza_pca.png")
print("comparacion_pca.png")
print("modelo_pca.joblib")

print("\nAnalisis PCA completado correctamente.")
