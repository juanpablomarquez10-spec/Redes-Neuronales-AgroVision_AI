from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------
# 1. Configuracion
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "modelos" / "mobilenetv2_tomate.keras"

TAMANO_IMAGEN = (180, 180)

CLASES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

NOMBRES = {
    "Tomato___Early_blight": "Tizon temprano",
    "Tomato___Late_blight": "Tizon tardio",
    "Tomato___healthy": "Hoja sana",
}


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print('python .\\src\\predecir_imagen.py "ruta_de_la_imagen"')
        sys.exit(1)

    ruta_imagen = Path(sys.argv[1]).expanduser()

    if not ruta_imagen.is_file():
        print(f"ERROR: no se encontro la imagen: {ruta_imagen}")
        sys.exit(1)

    if not RUTA_MODELO.is_file():
        print(f"ERROR: no se encontro el modelo: {RUTA_MODELO}")
        sys.exit(1)

    # --------------------------------------------------
    # 2. Preparar la imagen
    # --------------------------------------------------
    try:
        with Image.open(ruta_imagen) as imagen:
            imagen = imagen.convert("RGB")
            imagen = imagen.resize(TAMANO_IMAGEN)
            arreglo = np.asarray(imagen, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as error:
        print(f"ERROR: no se pudo leer la imagen: {error}")
        sys.exit(1)

    lote = np.expand_dims(arreglo, axis=0)

    # --------------------------------------------------
    # 3. Cargar modelo y predecir
    # --------------------------------------------------
    print("Cargando MobileNetV2...")
    modelo = tf.keras.models.load_model(RUTA_MODELO)

    if modelo.output_shape[-1] != len(CLASES):
        print("ERROR: la salida del modelo no coincide con las 3 clases.")
        sys.exit(1)

    probabilidades = modelo.predict(lote, verbose=0)[0]
    indice = int(np.argmax(probabilidades))

    print("\n--- RESULTADO DE AGROVISION_AI ---")
    print(f"Imagen: {ruta_imagen.name}")
    print(f"Prediccion: {NOMBRES[CLASES[indice]]}")
    print(f"Confianza del modelo: {probabilidades[indice] * 100:.2f}%")

    print("\nProbabilidades por categoria:")
    for clase, probabilidad in zip(CLASES, probabilidades):
        print(f"- {NOMBRES[clase]}: {probabilidad * 100:.2f}%")

    print(
        "\nNota: esta es una prediccion automatica, "
        "no un diagnostico agronomico definitivo."
    )


if __name__ == "__main__":
    main()