# Unidad 4 - Análisis no supervisado de ventas

## Descripción del proyecto

Este proyecto continúa el caso de estudio de la Unidad 2, relacionado con el análisis de ventas de productos electrónicos.

En esta unidad se aplican técnicas de aprendizaje no supervisado para identificar agrupamientos naturales en los registros de ventas, detectar posibles valores atípicos y reducir la dimensionalidad de los datos para su interpretación visual.

## Integrantes

- Juan Gerardo Galindo Pineda.
- Raúl de Jesús Martínez Cortez.
- Lidia Alejandra Guevara Ramírez.

## Objetivos

- Preparar los datos para análisis no supervisado.
- Aplicar K-Means para segmentar registros de venta.
- Optimizar la cantidad de clusters en K-Means.
- Aplicar DBSCAN para detectar agrupamientos y ruido.
- Comparar los modelos mediante métricas de evaluación.
- Reducir dimensiones con PCA para visualizar los resultados.

## Conjunto de datos

Se utilizó el archivo `datos/ventas_limpias.csv`, generado en la Unidad 2.

- Registros analizados: 115
- Variables originales: 15
- Variables utilizadas para clustering:
  - `cantidad`
  - `precio_unitario`
  - `total_venta`

## Tecnologías utilizadas

- Python 3.11
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

## Proceso realizado

### 1. Preparación de datos
Se seleccionaron tres variables numéricas: cantidad, precio unitario y total de venta. Posteriormente se verificó que no existieran valores faltantes y se aplicó estandarización con `StandardScaler`.

Archivos generados:
- `unidad4/resultados/datos_no_supervisados.csv`
- `unidad4/resultados/datos_escalados.csv`
- `unidad4/modelos/escalador.joblib`

### 2. K-Means inicial
Se entrenó un modelo K-Means con 3 clusters como prueba inicial.

Resultados:
- Clusters: 3
- Silhouette: 0.4755
- Calinski-Harabasz: 117.9474
- Davies-Bouldin: 0.7659

Archivos generados:
- `unidad4/resultados/clusters_kmeans_inicial.csv`
- `unidad4/resultados/metricas_kmeans_inicial.csv`
- `unidad4/resultados/kmeans_inicial.png`
- `unidad4/modelos/kmeans_inicial.joblib`

### 3. Optimización de K-Means
Se probaron diferentes cantidades de clusters y se seleccionó la mejor configuración con base en las métricas de agrupamiento.

Mejor configuración:
- Número de clusters: 8
- Silhouette: 0.6064
- Calinski-Harabasz: 246.7357
- Davies-Bouldin: 0.5552

Archivos generados:
- `unidad4/resultados/optimizacion_kmeans.csv`
- `unidad4/resultados/clusters_kmeans_optimizado.csv`
- `unidad4/resultados/optimizacion_kmeans.png`
- `unidad4/resultados/kmeans_optimizado.png`

### 4. DBSCAN
Se aplicó DBSCAN con configuración inicial y posteriormente se optimizó variando `eps` y `min_samples`.

#### DBSCAN inicial
- Clusters: 12
- Ruido: 10
- Cobertura: 91.30%
- Silhouette: 0.8256
- Calinski-Harabasz: 872.0835
- Davies-Bouldin: 0.2844

#### DBSCAN optimizado
- eps: 0.2
- min_samples: 3
- Clusters: 20
- Ruido: 6
- Cobertura: 94.78%
- Silhouette: 0.9504
- Calinski-Harabasz: 11626.8557
- Davies-Bouldin: 0.0782

Archivos generados:
- `unidad4/resultados/clusters_dbscan.csv`
- `unidad4/resultados/comparacion_dbscan.csv`
- `unidad4/resultados/optimizacion_dbscan.csv`
- `unidad4/resultados/comparacion_dbscan.png`

### 5. PCA
Se aplicó Análisis de Componentes Principales para reducir las variables a dos dimensiones y facilitar la visualización de los grupos.

Resultados:
- Componente 1: 60.45%
- Componente 2: 36.81%
- Varianza acumulada: 97.26%

Archivos generados:
- `unidad4/resultados/datos_pca.csv`
- `unidad4/resultados/componentes_pca.csv`
- `unidad4/resultados/varianza_pca.png`
- `unidad4/resultados/comparacion_pca.png`
- `unidad4/modelos/modelo_pca.joblib`

## Comparación final de modelos

| Modelo | Clusters | Ruido | Cobertura | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|--------|----------|-------|-----------|------------|-------------------|----------------|
| K-Means inicial | 3 | 0 | 1.0000 | 0.4755 | 117.9474 | 0.7659 |
| K-Means optimizado | 8 | 0 | 1.0000 | 0.6064 | 246.7357 | 0.5552 |
| DBSCAN inicial | 12 | 10 | 0.9130 | 0.8256 | 872.0835 | 0.2844 |
| DBSCAN optimizado | 20 | 6 | 0.9478 | 0.9504 | 11626.8557 | 0.0782 |

## Interpretación de resultados

K-Means permitió agrupar la totalidad de los registros, lo cual garantiza cobertura completa del conjunto de datos. Sin embargo, sus métricas fueron inferiores a las obtenidas por DBSCAN.

DBSCAN mostró mejor desempeño en términos de separación y calidad de agrupamiento, especialmente en la configuración optimizada. Además, permitió detectar valores considerados ruido o atípicos. No obstante, al excluir ruido, su cobertura no fue total, por lo que sus métricas deben interpretarse considerando esta diferencia metodológica.

PCA permitió representar la información en dos componentes principales conservando el 97.26% de la varianza total, lo que facilita la visualización de los grupos obtenidos por K-Means y DBSCAN.

## Conclusión

El análisis no supervisado permitió identificar patrones de comportamiento en las ventas a partir de variables numéricas clave. K-Means resultó útil para segmentar todos los registros en grupos definidos, mientras que DBSCAN ofreció mejor calidad de agrupamiento y capacidad para detectar ruido. PCA complementó el análisis al reducir la dimensionalidad sin pérdida significativa de información. En conjunto, estas técnicas demostraron ser herramientas valiosas para descubrir estructuras ocultas en los datos y apoyar la interpretación del comportamiento de ventas.

## Cómo reproducir

1. Ejecutar la preparación de datos:
   ```bash
   python unidad4/scripts/preparar_datos.py