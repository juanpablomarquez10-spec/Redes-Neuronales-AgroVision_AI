
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Todas las gráficas se guardarán dentro del proyecto.
BASE = Path(__file__).resolve().parent
SALIDA = BASE / "resultados_comparacion" / "graficas"
SALIDA.mkdir(parents=True, exist_ok=True)

# Resultados obtenidos durante la evaluación de los modelos.
modelos = ["CNN", "MobileNetV2"]
exactitudes = [71.11, 89.33]

categorias = [
    "Tizón temprano",
    "Tizón tardío",
    "Hoja sana",
]

# Valores F1 por categoría, obtenidos en la evaluación.
f1_cnn = [0.7065868263, 0.5357142857, 0.8304093567]
f1_mobilenet = [0.8513513514, 0.8904109589, 0.9358974359]

# Filas = categorías reales; columnas = predicciones.
matriz_cnn = np.array([
    [59, 5, 11],
    [31, 30, 14],
    [2, 2, 71],
])

matriz_mobilenet = np.array([
    [63, 5, 7],
    [9, 65, 1],
    [1, 1, 73],
])


def guardar_figura(nombre):
    ruta = SALIDA / nombre
    plt.tight_layout()
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada: {ruta}")


# 1. Comparación de exactitud.
plt.figure(figsize=(8, 5))
barras = plt.bar(modelos, exactitudes)
plt.title("Comparación de exactitud de los modelos")
plt.ylabel("Exactitud (%)")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.25)

for barra, valor in zip(barras, exactitudes):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 1,
        f"{valor:.2f}%",
        ha="center",
    )

guardar_figura("01_comparacion_exactitud.png")


# 2. Comparación del F1 por categoría.
x = np.arange(len(categorias))
ancho = 0.36

plt.figure(figsize=(10, 6))
b1 = plt.bar(x - ancho / 2, f1_cnn, ancho, label="CNN")
b2 = plt.bar(
    x + ancho / 2,
    f1_mobilenet,
    ancho,
    label="MobileNetV2",
)

plt.title("Comparación del F1-score por categoría")
plt.ylabel("F1-score")
plt.xticks(x, categorias, rotation=10)
plt.ylim(0, 1.08)
plt.legend()
plt.grid(axis="y", alpha=0.25)

for barras in (b1, b2):
    for barra in barras:
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.015,
            f"{barra.get_height():.2f}",
            ha="center",
            fontsize=8,
        )

guardar_figura("02_f1_por_categoria.png")


# Función para visualizar una matriz de confusión.
def graficar_matriz(matriz, titulo, archivo):
    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(matriz, cmap="Blues")

    ax.set_title(titulo)
    ax.set_xlabel("Categoría predicha")
    ax.set_ylabel("Categoría real")
    ax.set_xticks(range(len(categorias)))
    ax.set_yticks(range(len(categorias)))
    ax.set_xticklabels(categorias, rotation=20, ha="right")
    ax.set_yticklabels(categorias)

    limite = matriz.max() / 2

    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            valor = matriz[fila, columna]
            color = "white" if valor > limite else "black"
            ax.text(
                columna,
                fila,
                str(valor),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )

    fig.colorbar(imagen, ax=ax, label="Número de imágenes")
    guardar_figura(archivo)


# 3. Matriz de confusión de CNN.
graficar_matriz(
    matriz_cnn,
    "Matriz de confusión - CNN",
    "03_matriz_confusion_cnn.png",
)

# 4. Matriz de confusión de MobileNetV2.
graficar_matriz(
    matriz_mobilenet,
    "Matriz de confusión - MobileNetV2",
    "04_matriz_confusion_mobilenetv2.png",
)

print("\nProceso terminado. Se generaron las cuatro gráficas.")
