from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_divididos"
RESULTADOS_DIR = BASE_DIR / "resultados"
MODELOS_DIR = BASE_DIR / "modelos"

TAMANO_IMAGEN = (180, 180)
TAMANO_LOTE = 32
SEMILLA = 42
EPOCAS = 20

tf.keras.utils.set_random_seed(SEMILLA)

# --------------------------------------------------
# 2. Verificar los datos
# --------------------------------------------------
for conjunto in ["entrenamiento", "validacion", "prueba"]:
    if not (DATA_DIR / conjunto).is_dir():
        raise FileNotFoundError(
            f"No existe el conjunto: {DATA_DIR / conjunto}"
        )

RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
MODELOS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 3. Cargar imágenes
# --------------------------------------------------
def cargar_conjunto(nombre, mezclar):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / nombre,
        image_size=TAMANO_IMAGEN,
        batch_size=TAMANO_LOTE,
        label_mode="int",
        shuffle=mezclar,
        seed=SEMILLA
    )

train_ds = cargar_conjunto("entrenamiento", True)
val_ds = cargar_conjunto("validacion", False)
test_ds = cargar_conjunto("prueba", False)

clases = train_ds.class_names
numero_clases = len(clases)

if numero_clases != 3:
    raise ValueError(
        f"Se esperaban 3 clases y se encontraron {numero_clases}: {clases}"
    )

if val_ds.class_names != clases or test_ds.class_names != clases:
    raise ValueError("Las clases no coinciden entre los conjuntos.")

print("\nCategorías:", clases)
print("TensorFlow:", tf.__version__)

# Optimización de entrada
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# 4. Construir la red neuronal convolucional (CNN)
# --------------------------------------------------
aumento = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="aumento_datos")

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(180, 180, 3)),
    aumento,
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(numero_clases, activation="softmax")
])

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# --------------------------------------------------
# 5. Entrenar y guardar el mejor modelo
# --------------------------------------------------
ruta_modelo = MODELOS_DIR / "cnn_tomate.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ruta_modelo),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

historial = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCAS,
    callbacks=callbacks
)

# Recuperar el mejor modelo guardado
modelo = tf.keras.models.load_model(ruta_modelo)

# --------------------------------------------------
# 6. Evaluación final en el conjunto de prueba
# --------------------------------------------------
perdida, exactitud = modelo.evaluate(test_ds, verbose=1)

etiquetas_reales = np.concatenate([
    etiquetas.numpy()
    for _, etiquetas in test_ds
])

probabilidades = modelo.predict(test_ds, verbose=1)
predicciones = np.argmax(probabilidades, axis=1)

informe = classification_report(
    etiquetas_reales,
    predicciones,
    labels=list(range(numero_clases)),
    target_names=clases,
    output_dict=True,
    zero_division=0
)

print("\nRESULTADOS DEL CONJUNTO DE PRUEBA")
print(f"Exactitud: {exactitud:.4f} ({exactitud * 100:.2f} %)")
print(f"Pérdida: {perdida:.4f}")

print("\nInforme de clasificación:")
print(classification_report(
    etiquetas_reales,
    predicciones,
    labels=list(range(numero_clases)),
    target_names=clases,
    zero_division=0
))

# --------------------------------------------------
# 7. Guardar métricas y resultados
# --------------------------------------------------
with (RESULTADOS_DIR / "metricas_prueba.json").open(
    "w", encoding="utf-8"
) as archivo:
    json.dump({
        "tensorflow": tf.__version__,
        "clases": clases,
        "tamano_imagen": list(TAMANO_IMAGEN),
        "semilla": SEMILLA,
        "epocas_maximas": EPOCAS,
        "exactitud_prueba": float(exactitud),
        "perdida_prueba": float(perdida),
        "informe_clasificacion": informe
    }, archivo, indent=4, ensure_ascii=False)

with (RESULTADOS_DIR / "historial_entrenamiento.csv").open(
    "w", newline="", encoding="utf-8"
) as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["epoca", "exactitud", "val_exactitud",
                       "perdida", "val_perdida"])

    for i in range(len(historial.history["loss"])):
        escritor.writerow([
            i + 1,
            historial.history["accuracy"][i],
            historial.history["val_accuracy"][i],
            historial.history["loss"][i],
            historial.history["val_loss"][i]
        ])

# Matriz de confusión
matriz = confusion_matrix(
    etiquetas_reales,
    predicciones,
    labels=list(range(numero_clases))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=clases
)
disp.plot(xticks_rotation=30, values_format="d")
plt.title("Matriz de confusión - Prueba")
plt.tight_layout()
plt.savefig(
    RESULTADOS_DIR / "matriz_confusion.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# Curvas de entrenamiento y validación
fig, ejes = plt.subplots(1, 2, figsize=(12, 4))

ejes[0].plot(historial.history["accuracy"], label="Entrenamiento")
ejes[0].plot(historial.history["val_accuracy"], label="Validación")
ejes[0].set_title("Exactitud por época")
ejes[0].set_xlabel("Época")
ejes[0].set_ylabel("Exactitud")
ejes[0].legend()

ejes[1].plot(historial.history["loss"], label="Entrenamiento")
ejes[1].plot(historial.history["val_loss"], label="Validación")
ejes[1].set_title("Pérdida por época")
ejes[1].set_xlabel("Época")
ejes[1].set_ylabel("Pérdida")
ejes[1].legend()

plt.tight_layout()
plt.savefig(
    RESULTADOS_DIR / "curvas_entrenamiento.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nENTRENAMIENTO Y EVALUACIÓN COMPLETADOS")
print(f"Modelo: {ruta_modelo}")
print(f"Resultados: {RESULTADOS_DIR}")
print("Archivos generados:")
print("- metricas_prueba.json")
print("- historial_entrenamiento.csv")
print("- matriz_confusion.png")
print("- curvas_entrenamiento.png")