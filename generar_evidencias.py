from pathlib import Path
import json
import shutil
from datetime import datetime

BASE = Path(__file__).resolve().parent
DESTINO = BASE / "evidencias"
DESTINO.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("GENERADOR DE EVIDENCIAS - AGROVISION AI")
print("=" * 55)

# 1. Resumen del conjunto de datos
carpeta_datos = BASE / "data_divididos"
resumen = {
    "fecha_generacion": datetime.now().isoformat(timespec="seconds"),
    "proyecto": "AgroVision AI",
    "dataset": {}
}

if carpeta_datos.exists():
    alternativas = {
        "entrenamiento": "train",
        "validacion": "val",
        "prueba": "test"
    }

    for particion in ["entrenamiento", "validacion", "prueba"]:
        carpeta = carpeta_datos / particion
        if not carpeta.exists():
            carpeta = carpeta_datos / alternativas[particion]

        if carpeta.exists():
            resumen["dataset"][particion] = {}

            for clase in sorted(carpeta.iterdir()):
                if clase.is_dir():
                    cantidad = sum(
                        1 for f in clase.iterdir()
                        if f.is_file() and f.suffix.lower()
                        in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    )
                    resumen["dataset"][particion][clase.name] = cantidad

    (DESTINO / "resumen_dataset.json").write_text(
        json.dumps(resumen, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    print("[OK] Resumen del conjunto de datos")
else:
    print("[AVISO] No se encontró data_divididos")

# 2. Copiar los resultados originales de entrenamiento
recursos = [
    ("resultados", "CNN"),
    ("resultados_mobilenet", "MobileNetV2")
]

for carpeta_nombre, modelo in recursos:
    origen = BASE / carpeta_nombre
    if not origen.exists():
        print(f"[AVISO] No existe {carpeta_nombre}")
        continue

    destino_modelo = DESTINO / modelo
    destino_modelo.mkdir(parents=True, exist_ok=True)

    for archivo in origen.iterdir():
        if archivo.is_file() and archivo.suffix.lower() in {
            ".json", ".png", ".csv", ".txt"
        }:
            shutil.copy2(archivo, destino_modelo / archivo.name)
            print(f"[OK] {modelo}/{archivo.name}")

# 3. Copiar las evaluaciones detalladas actuales
evaluaciones = [
    ("CNN", BASE / "resultados_cnn" / "evaluacion_cnn.json"),
    ("MobileNetV2", BASE / "resultados_mobilenet" / "evaluacion_mobilenet.json")
]

comparacion = []

for modelo, archivo in evaluaciones:
    if not archivo.exists():
        print(f"[AVISO] Falta evaluación de {modelo}: {archivo}")
        continue

    try:
        datos = json.loads(archivo.read_text(encoding="utf-8-sig"))
        destino_modelo = DESTINO / modelo
        destino_modelo.mkdir(parents=True, exist_ok=True)

        (destino_modelo / "evaluacion_detallada.json").write_text(
            json.dumps(datos, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

        total = int(datos["imagenes_evaluadas"])
        exactitud = float(datos["exactitud"])
        correctas = round(total * exactitud)

        comparacion.append({
            "modelo": modelo,
            "imagenes_evaluadas": total,
            "predicciones_correctas": correctas,
            "predicciones_incorrectas": total - correctas,
            "exactitud_porcentaje": round(exactitud * 100, 2)
        })

        print(f"[OK] Evaluación detallada: {modelo}")

    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[AVISO] No se pudo procesar {modelo}: {error}")

# 4. Guardar comparación y resumen Markdown
(DESTINO / "comparacion_modelos.json").write_text(
    json.dumps(comparacion, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

(DESTINO / "comparacion_modelos.csv").write_text(
    "modelo,imagenes_evaluadas,predicciones_correctas,"
    "predicciones_incorrectas,exactitud_porcentaje\n"
    + "".join(
        f'{x["modelo"]},{x["imagenes_evaluadas"]},'
        f'{x["predicciones_correctas"]},'
        f'{x["predicciones_incorrectas"]},'
        f'{x["exactitud_porcentaje"]:.2f}\n'
        for x in comparacion
    ),
    encoding="utf-8-sig"
)

lineas = [
    "# Evidencias del proyecto AgroVision AI",
    "",
    f"Fecha de generación: {datetime.now():%Y-%m-%d %H:%M:%S}",
    "",
    "## Conjunto de datos",
    ""
]

if resumen["dataset"]:
    for particion, clases in resumen["dataset"].items():
        total = sum(clases.values())
        lineas.append(f"### {particion}: {total} imágenes")
        for clase, cantidad in clases.items():
            lineas.append(f"- {clase}: {cantidad} imágenes")
        lineas.append("")
else:
    lineas.append("No se pudo generar el resumen del conjunto de datos.")

lineas += ["## Evaluación de modelos", ""]

for resultado in comparacion:
    lineas.append(
        f'- **{resultado["modelo"]}:** '
        f'{resultado["exactitud_porcentaje"]:.2f}% de exactitud; '
        f'{resultado["predicciones_correctas"]} aciertos de '
        f'{resultado["imagenes_evaluadas"]} imágenes.'
    )

lineas += [
    "",
    "Las evaluaciones detalladas se guardan en las carpetas CNN y "
    "MobileNetV2. Los modelos no se vuelven a entrenar con este script.",
    ""
]

(DESTINO / "informe_evidencias.md").write_text(
    "\n".join(lineas),
    encoding="utf-8"
)

print("\nProceso terminado.")
print(f"Carpeta de evidencias: {DESTINO}")
