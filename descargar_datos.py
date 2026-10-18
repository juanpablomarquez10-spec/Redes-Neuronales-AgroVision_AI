from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from io import BytesIO
from collections import Counter
import requests

# Ruta principal del proyecto
RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATOS = RAIZ / "data"

# Clases que utilizaremos
CLASES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
]

REPOSITORIO = "lorenzoxi/tomato-leaves-dataset"
PARTICION = "train"
MAX_IMAGENES_POR_CLASE = 500

api = HfApi()

print("Consultando imágenes disponibles...")
archivos = api.list_repo_files(
    repo_id=REPOSITORIO,
    repo_type="dataset"
)

# Seleccionar únicamente las imágenes de entrenamiento
seleccionados = {
    clase: [
        archivo for archivo in archivos
        if archivo.startswith(f"dataset-tomatoes/{PARTICION}/{clase}/")
        and archivo.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:MAX_IMAGENES_POR_CLASE]
    for clase in CLASES
}

for clase, lista in seleccionados.items():
    print(f"{clase}: {len(lista)} imágenes disponibles para descargar")

print("\nIniciando descarga...")

resumen = Counter()

for clase, lista in seleccionados.items():
    destino = CARPETA_DATOS / clase
    destino.mkdir(parents=True, exist_ok=True)

    for numero, archivo in enumerate(lista, start=1):
        try:
            # Descargar el archivo usando la biblioteca oficial.
            ruta_cache = hf_hub_download(
                repo_id=REPOSITORIO,
                filename=archivo,
                repo_type="dataset",
            )

            # Abrir y guardar con un nombre seguro para Windows.
            imagen = Image.open(ruta_cache).convert("RGB")
            nombre = f"imagen_{numero:04d}.jpg"
            imagen.save(destino / nombre, format="JPEG")

            resumen[clase] += 1

        except Exception as error:
            print(f"No se pudo procesar {archivo}: {error}")

    print(f"Procesada clase {clase}: {resumen[clase]} imágenes guardadas")

print("\nRESUMEN FINAL")
for clase in CLASES:
    print(f"{clase}: {resumen[clase]} imágenes")

print(f"Total de imágenes guardadas: {sum(resumen.values())}")
print(f"Ubicación: {CARPETA_DATOS.resolve()}")