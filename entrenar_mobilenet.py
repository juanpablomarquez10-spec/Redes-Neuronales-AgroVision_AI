from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# --------------------------------------------------
# 1. Configuracion
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_divididos"
RESULTADOS_DIR = BASE_DIR / "resultados_mobilenet"
MODELOS_DIR = BASE_DIR / "modelos"

TAMANO_IMAGEN = (180, 180)
TAMANO_LOTE = 32
SEMILLA = 42
EPOCAS = 15

tf.keras.utils.set_random_seed(SEMILLA)

RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
MODELOS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Cargar los conjuntos de datos
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

if len(clases) != 3:
    raise ValueError(f"Se esperaban 3 clases: {clases}")

if val_ds.class_names != clases or test_ds.class_names != clases:
    raise ValueError("Las clases no coinciden entre conjuntos.")

print("\nClases:", clases)
print("TensorFlow:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# 3. Modelo MobileNetV2 preentrenado
# --------------------------------------------------
# Se intenta cargar ImageNet con pesos preentrenados.
# Si no hay internet, el programa se detendra con un
# error explicativo; no se sustituiran silenciosamente.
base = tf.keras.applications.MobileNetV2(
    input_shape=(180, 180, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False

entrada = tf.keras.Input(shape=(180, 180, 3))

# MobileNetV2 espera valores normalizados entre -1 y 1.
x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(entrada)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
salida = tf.keras.layers.Dense(len(clases), activation="softmax")(x)

modelo = tf.keras.Model(entrada, salida, name="mobilenetv2_tomate")

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

modelo.summary()

# --------------------------------------------------
# 4. Entrenamiento
# --------------------------------------------------
ruta_modelo = MODELOS_DIR / "mobilenetv2_tomate.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
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

modelo = tf.keras.models.load_model(ruta_modelo)

# --------------------------------------------------
# 5. Evaluacion final en prueba
# --------------------------------------------------
perdida, exactitud = modelo.evaluate(test_ds, verbose=1)

reales = np.concatenate([
    etiquetas.numpy() for _, etiquetas in test_ds
])

probabilidades = modelo.predict(test_ds, verbose=1)
predicciones = np.argmax(probabilidades, axis=1)

informe_texto = classification_report(
    reales,
    predicciones,
    labels=list(range(len(clases))),
    target_names=clases,
    zero_division=0
)

informe = classification_report(
    reales,
    predicciones,
    labels=list(range(len(clases))),
    target_names=clases,
    output_dict=True,
    zero_division=0
)

print("\nRESULTADOS MOBILENETV2")
print(f"Exactitud de prueba: {exactitud:.4f} ({exactitud * 100:.2f} %)")
print(f"Perdida de prueba: {perdida:.4f}")
print(informe_texto)

with (RESULTADOS_DIR / "metricas_prueba.json").open(
    "w", encoding="utf-8"
) as archivo:
    json.dump({
        "modelo": "MobileNetV2",
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
    escritor.writerow([
        "epoca", "exactitud", "val_exactitud",
        "perdida", "val_perdida"
    ])

    for i in range(len(historial.history["loss"])):
        escritor.writerow([
            i + 1,
            historial.history["accuracy"][i],
            historial.history["val_accuracy"][i],
            historial.history["loss"][i],
            historial.history["val_loss"][i]
        ])

matriz = confusion_matrix(
    reales, predicciones, labels=list(range(len(clases)))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=clases
)
disp.plot(xticks_rotation=30, values_format="d")
plt.title("Matriz de confusion - MobileNetV2")
plt.tight_layout()
plt.savefig(
    RESULTADOS_DIR / "matriz_confusion.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

fig, ejes = plt.subplots(1, 2, figsize=(12, 4))

ejes[0].plot(historial.history["accuracy"], label="Entrenamiento")
ejes[0].plot(historial.history["val_accuracy"], label="Validacion")
ejes[0].set_title("Exactitud por epoca")
ejes[0].set_xlabel("Epoca")
ejes[0].set_ylabel("Exactitud")
ejes[0].legend()

ejes[1].plot(historial.history["loss"], label="Entrenamiento")
ejes[1].plot(historial.history["val_loss"], label="Validacion")
ejes[1].set_title("Perdida por epoca")
ejes[1].set_xlabel("Epoca")
ejes[1].set_ylabel("Perdida")
ejes[1].legend()

plt.tight_layout()
plt.savefig(
    RESULTADOS_DIR / "curvas_entrenamiento.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()

print("\nEXPERIMENTO MOBILENETV2 COMPLETADO")
print("Modelo:", ruta_modelo)
print("Resultados:", RESULTADOS_DIR)