from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTADOS = ROOT / "unidad3" / "resultados"

# Cargar resultados de los modelos
ridge = pd.read_csv(RESULTADOS / "metricas_regresion.csv")
forest = pd.read_csv(RESULTADOS / "metricas_random_forest.csv")
clasificacion = pd.read_csv(RESULTADOS / "metricas_clasificacion.csv")

# Consolidar resultados de regresion
regresion = pd.concat([ridge, forest], ignore_index=True)

regresion.to_csv(
    RESULTADOS / "comparacion_regresion.csv", index=False
)

print("===== COMPARACION DE REGRESION =====")
print(regresion.to_string(index=False))

print("\n===== COMPARACION DE CLASIFICACION =====")
print(clasificacion.to_string(index=False))

# Graficas de regresion
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

regresion.set_index("Modelo")[["MAE", "RMSE"]].plot(
    kind="bar", ax=axes[0], color=["steelblue", "orange"]
)

axes[0].set_title("Comparacion de errores")
axes[0].set_ylabel("Importe en MXN")
axes[0].tick_params(axis="x", rotation=25)

regresion.set_index("Modelo")["R2"].plot(
    kind="bar", ax=axes[1], color="teal"
)

axes[1].set_title("Coeficiente de determinacion R2")
axes[1].tick_params(axis="x", rotation=25)
axes[1].axhline(0, color="black", linewidth=0.8)

plt.tight_layout()
plt.savefig(RESULTADOS / "comparacion_regresion.png", dpi=250)
plt.close()

# Grafica de clasificacion
clasificacion.set_index("Modelo")[
    ["Accuracy", "Precision", "Recall", "F1"]
].plot(kind="bar", figsize=(13, 7))

plt.title("Comparacion de modelos de clasificacion")
plt.ylabel("Puntuacion")
plt.ylim(0, 1.1)
plt.xticks(rotation=20)
plt.legend()
plt.tight_layout()

plt.savefig(
    RESULTADOS / "comparacion_clasificacion.png", dpi=250
)
plt.close()

# Mejores resultados observados
mejor_regresion = regresion.loc[regresion["MAE"].idxmin()]
mejor_clasificacion = clasificacion.loc[
    clasificacion["F1"].idxmax()
]

print("\n===== MEJORES RESULTADOS EN PRUEBA =====")
print("Regresion:", mejor_regresion["Modelo"])
print("Clasificacion:", mejor_clasificacion["Modelo"])

print("\nGraficas comparativas generadas correctamente.")