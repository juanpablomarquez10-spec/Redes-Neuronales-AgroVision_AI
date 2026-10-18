from pathlib import Path
from collections import Counter
import shutil
import csv

from sklearn.model_selection import train_test_split

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data_divididos"

# Configuración
SEMILLA = 42
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"No se encuentra el directorio: {DATA_DIR}"
        )

    # Leer las carpetas de categorías
    clases = sorted(
        carpeta.name
        for carpeta in DATA_DIR.iterdir()
        if carpeta.is_dir()
    )

    if len(clases) < 2:
        raise ValueError(
            "Se necesitan al menos dos carpetas de categorías."
        )

    # No sobrescribir una división anterior
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Ya existe {OUTPUT_DIR}. "
            "Revisa esa carpeta antes de volver a ejecutar."
        )

    imagenes_por_clase = {}

    for clase in clases:
        archivos = sorted(
            archivo
            for archivo in (DATA_DIR / clase).rglob("*")
            if archivo.is_file()
            and archivo.suffix.lower() in EXTENSIONES
        )

        if len(archivos) < 3:
            raise ValueError(
                f"La categoría {clase} tiene "
                f"muy pocas imágenes: {len(archivos)}."
            )

        imagenes_por_clase[clase] = archivos

    # Primera división: 70 % entrenamiento y 30 % temporal.
    entrenamiento = []
    temporal = []

    for clase, archivos in imagenes_por_clase.items():
        train, temp = train_test_split(
            archivos,
            test_size=0.30,
            random_state=SEMILLA,
            shuffle=True
        )

        entrenamiento.extend((archivo, clase) for archivo in train)
        temporal.extend((archivo, clase) for archivo in temp)

    # Segunda división: repartir el conjunto temporal
    # en validación y prueba, aproximadamente 15 % cada uno.
    temporal_paths = [archivo for archivo, _ in temporal]
    temporal_clases = [clase for _, clase in temporal]

    validacion_paths, prueba_paths, validacion_clases, prueba_clases = (
        train_test_split(
            temporal_paths,
            temporal_clases,
            test_size=0.50,
            random_state=SEMILLA,
            stratify=temporal_clases
        )
    )

    conjuntos = {
        "entrenamiento": entrenamiento,
        "validacion": list(zip(validacion_paths, validacion_clases)),
        "prueba": list(zip(prueba_paths, prueba_clases)),
    }

    # Copiar archivos y comprobar que no se repitan.
    archivos_origen = set()
    resumen = []

    for nombre, elementos in conjuntos.items():
        for archivo, clase in elementos:
            if archivo in archivos_origen:
                raise RuntimeError(
                    f"Imagen asignada más de una vez: {archivo}"
                )
            archivos_origen.add(archivo)

            destino = OUTPUT_DIR / nombre / clase
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archivo, destino / archivo.name)

        conteo = Counter(clase for _, clase in elementos)

        print(f"\n{nombre.upper()}: {len(elementos)} imágenes")
        for clase in clases:
            cantidad = conteo[clase]
            print(f"  {clase}: {cantidad}")
            resumen.append({
                "conjunto": nombre,
                "clase": clase,
                "cantidad": cantidad
            })

    # Guardar resultados para documentar el experimento.
    with (OUTPUT_DIR / "resumen_division.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo_csv:
        escritor = csv.DictWriter(
            archivo_csv,
            fieldnames=["conjunto", "clase", "cantidad"]
        )
        escritor.writeheader()
        escritor.writerows(resumen)

    total = sum(len(elementos) for elementos in conjuntos.values())

    print("\nDIVISIÓN COMPLETADA")
    print(f"Total de imágenes: {total}")
    print(f"Carpeta de salida: {OUTPUT_DIR}")
    print("Distribución objetivo: 70 % / 15 % / 15 %")
    print("Semilla aleatoria: 42")
    print("Resumen guardado en resumen_division.csv")


if __name__ == "__main__":
    main()