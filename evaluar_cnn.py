
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# =====================================================
# RUTAS Y CONFIGURACIÓN
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "modelos" / "cnn_tomate.keras"
RUTA_PRUEBA = BASE_DIR / "data_divididos" / "prueba"
RUTA_RESULTADOS = BASE_DIR / "resultados_cnn"

TAMANO_IMAGEN = (180, 180)

CLASES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

NOMBRES = [
    "Tizón temprano",
    "Tizón tardío",
    "Hoja sana",
]


def main():

    print("=" * 60)
    print("AGROVISION IA - EVALUACIÓN DE CNN")
    print("=" * 60)

    # Verificar rutas.
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo CNN:\n{RUTA_MODELO}"
        )

    if not RUTA_PRUEBA.is_dir():
        raise FileNotFoundError(
            f"No se encontró la carpeta de prueba:\n{RUTA_PRUEBA}"
        )

    for clase in CLASES:
        if not (RUTA_PRUEBA / clase).is_dir():
            raise FileNotFoundError(
                f"Falta la carpeta de la categoría: {clase}"
            )

    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # Cargar el modelo.
    print("\nCargando modelo CNN...")
    modelo = tf.keras.models.load_model(RUTA_MODELO)
    print("Modelo cargado correctamente.")

    # Leer las imágenes.
    imagenes = []
    etiquetas_reales = []

    extensiones_validas = {".jpg", ".jpeg", ".png"}

    print("\nCargando imágenes de prueba...")

    for indice_clase, clase in enumerate(CLASES):

        carpeta = RUTA_PRUEBA / clase

        archivos = sorted(
            archivo
            for archivo in carpeta.iterdir()
            if archivo.is_file()
            and archivo.suffix.lower() in extensiones_validas
        )

        print(f"{clase}: {len(archivos)} imágenes")

        for archivo in archivos:

            imagen = tf.keras.utils.load_img(
                archivo,
                target_size=TAMANO_IMAGEN,
                color_mode="rgb",
            )

            arreglo = tf.keras.utils.img_to_array(imagen)

            imagenes.append(arreglo)
            etiquetas_reales.append(indice_clase)

    if not imagenes:
        raise ValueError("No se encontraron imágenes válidas.")

    X_prueba = np.asarray(imagenes, dtype=np.float32)
    y_real = np.asarray(etiquetas_reales, dtype=np.int32)

    print(f"\nTotal de imágenes: {len(X_prueba)}")
    print("\nGenerando predicciones...")

    probabilidades = modelo.predict(
        X_prueba,
        batch_size=32,
        verbose=1,
    )

    if (
        probabilidades.ndim != 2
        or probabilidades.shape[1] != len(CLASES)
    ):
        raise ValueError(
            "La salida del modelo no coincide con las tres categorías. "
            f"Forma recibida: {probabilidades.shape}"
        )

    if not np.all(np.isfinite(probabilidades)):
        raise ValueError("El modelo produjo valores no válidos.")

    y_pred = np.argmax(probabilidades, axis=1)

    # Calcular métricas.
    exactitud = accuracy_score(y_real, y_pred)

    informe = classification_report(
        y_real,
        y_pred,
        labels=[0, 1, 2],
        target_names=NOMBRES,
        output_dict=True,
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=[0, 1, 2],
    )

    print("\n" + "=" * 60)
    print("RESULTADOS DE EVALUACIÓN - CNN")
    print("=" * 60)

    print(f"Imágenes evaluadas: {len(y_real)}")
    print(f"Exactitud global: {exactitud * 100:.2f}%")

    print("\nInforme por categoría:")

    print(
        classification_report(
            y_real,
            y_pred,
            labels=[0, 1, 2],
            target_names=NOMBRES,
            zero_division=0,
        )
    )

    print("Matriz de confusión:")
    print(matriz)

    # Guardar métricas en JSON.
    datos = {
        "modelo": RUTA_MODELO.name,
        "imagenes_evaluadas": int(len(y_real)),
        "exactitud": float(exactitud),
        "metricas_por_categoria": informe,
        "matriz_confusion": matriz.tolist(),
        "orden_clases": CLASES,
        "nombres_clases": NOMBRES,
        "tamano_imagen": list(TAMANO_IMAGEN),
    }

    ruta_json = RUTA_RESULTADOS / "evaluacion_cnn.json"

    with ruta_json.open("w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            ensure_ascii=False,
            indent=4,
        )

    # Generar matriz de confusión.
    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=NOMBRES,
    )

    visualizacion.plot(
        values_format="d",
        xticks_rotation=20,
    )

    plt.title("Matriz de confusión - CNN")
    plt.tight_layout()

    ruta_grafica = RUTA_RESULTADOS / "matriz_confusion_cnn.png"

    plt.savefig(
        ruta_grafica,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print("\nArchivos guardados:")
    print(ruta_json)
    print(ruta_grafica)

    print("\nEvaluación CNN finalizada correctamente.")


if __name__ == "__main__":
    main()
