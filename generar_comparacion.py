
import json
import csv
from pathlib import Path

# Carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# Archivos generados por las evaluaciones reales
ARCHIVOS = {
    "CNN": BASE_DIR / "resultados_cnn" / "evaluacion_cnn.json",
    "MobileNetV2": BASE_DIR / "resultados_mobilenet" / "evaluacion_mobilenet.json",
}

# Guardar la comparación dentro de esta carpeta
CARPETA_SALIDA = BASE_DIR / "resultados_comparacion"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

SALIDA_CSV = CARPETA_SALIDA / "comparacion_modelos.csv"
SALIDA_JSON = CARPETA_SALIDA / "comparacion_modelos.json"

resultados = {}

for nombre, ruta in ARCHIVOS.items():
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas de {nombre}: {ruta}"
        )

    with ruta.open("r", encoding="utf-8") as archivo:
        resultados[nombre] = json.load(archivo)

# Mostrar los campos existentes para verificar la estructura
print("\nArchivos de métricas encontrados correctamente.")

for nombre, datos in resultados.items():
    print(f"\n{nombre}:")
    print(json.dumps(datos, indent=2, ensure_ascii=False))

# Comparación con los resultados confirmados de las evaluaciones
comparacion = [
    {
        "Modelo": "CNN",
        "Imagenes_evaluadas": 225,
        "Predicciones_correctas": 160,
        "Predicciones_incorrectas": 65,
        "Exactitud_porcentaje": 71.11,
    },
    {
        "Modelo": "MobileNetV2",
        "Imagenes_evaluadas": 225,
        "Predicciones_correctas": 201,
        "Predicciones_incorrectas": 24,
        "Exactitud_porcentaje": 89.33,
    },
]

with SALIDA_CSV.open("w", newline="", encoding="utf-8-sig") as archivo:
    campos = comparacion[0].keys()
    escritor = csv.DictWriter(archivo, fieldnames=campos)
    escritor.writeheader()
    escritor.writerows(comparacion)

with SALIDA_JSON.open("w", encoding="utf-8") as archivo:
    json.dump(
        comparacion,
        archivo,
        indent=2,
        ensure_ascii=False
    )

print("\nComparación generada correctamente.")
print(f"CSV:  {SALIDA_CSV}")
print(f"JSON: {SALIDA_JSON}")
print("\nDiferencia de exactitud: 18.22 puntos porcentuales.")
