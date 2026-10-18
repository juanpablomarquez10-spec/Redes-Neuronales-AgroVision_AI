from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk, UnidentifiedImageError


# =========================================================
# CONFIGURACIÓN Y RUTAS DEL PROYECTO
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = (
    BASE_DIR / "modelos" / "mobilenetv2_tomate.keras"
)

TAMANO_IMAGEN = (180, 180)

# El orden debe coincidir exactamente con el utilizado
# durante el entrenamiento del modelo.
CLASES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

NOMBRES = {
    "Tomato___Early_blight": "Tizón temprano",
    "Tomato___Late_blight": "Tizón tardío",
    "Tomato___healthy": "Hoja sana",
}


class AgroVisionApp:

    def __init__(self, root):
        self.root = root

        self.root.title(
            "AgroVision IA - Clasificador de hojas de tomate"
        )
        self.root.geometry("720x700")
        self.root.minsize(600, 600)

        self.modelo = None
        self.imagen_tk = None

        # ---------------- INTERFAZ ----------------

        contenedor = ttk.Frame(root, padding=20)
        contenedor.pack(fill="both", expand=True)

        ttk.Label(
            contenedor,
            text="AgroVision IA",
            font=("Segoe UI", 24, "bold")
        ).pack(pady=(0, 4))

        ttk.Label(
            contenedor,
            text=(
                "Clasificación de enfermedades "
                "en hojas de tomate"
            ),
            font=("Segoe UI", 11)
        ).pack(pady=(0, 18))

        self.boton = ttk.Button(
            contenedor,
            text="Seleccionar fotografía",
            command=self.seleccionar_imagen
        )
        self.boton.pack(pady=6)

        self.etiqueta_imagen = ttk.Label(
            contenedor,
            text="Selecciona una imagen JPG, JPEG o PNG",
            anchor="center"
        )
        self.etiqueta_imagen.pack(
            fill="both",
            expand=True,
            pady=12
        )

        self.resultado = ttk.Label(
            contenedor,
            text="La predicción aparecerá aquí",
            font=("Segoe UI", 15, "bold"),
            anchor="center",
            justify="center",
            wraplength=620
        )
        self.resultado.pack(pady=12)

        self.detalle = ttk.Label(
            contenedor,
            text="",
            font=("Segoe UI", 11),
            justify="left"
        )
        self.detalle.pack(pady=8)

        ttk.Label(
            contenedor,
            text=(
                "Herramienta experimental. "
                "No reemplaza un diagnóstico agronómico."
            ),
            wraplength=620,
            justify="center"
        ).pack(pady=(12, 0))

    # =====================================================
    # SELECCIONAR Y PROCESAR IMAGEN
    # =====================================================

    def seleccionar_imagen(self):

        ruta = filedialog.askopenfilename(
            title="Seleccionar una hoja de tomate",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not ruta:
            return

        try:
            with Image.open(ruta) as original:
                imagen = original.convert("RGB")

        except (OSError, UnidentifiedImageError) as error:
            messagebox.showerror(
                "Error de imagen",
                f"No fue posible abrir la imagen.\n\n{error}"
            )
            return

        # Mostrar una vista previa sin modificar
        # las dimensiones de la imagen original.
        vista = imagen.copy()
        vista.thumbnail((420, 300))

        self.imagen_tk = ImageTk.PhotoImage(vista)

        self.etiqueta_imagen.configure(
            image=self.imagen_tk,
            text=""
        )

        # Preparar la imagen con el tamaño esperado.
        imagen_modelo = imagen.resize(TAMANO_IMAGEN)

        arreglo = np.asarray(
            imagen_modelo,
            dtype=np.float32
        )

        lote = np.expand_dims(arreglo, axis=0)

        # Liberar la imagen original de la memoria
        # cuando ya no sea necesaria.
        imagen.close()

        self.predecir(lote)

    # =====================================================
    # CARGAR MODELO Y PREDECIR
    # =====================================================

    def predecir(self, lote):

        try:
            # Cargar el modelo solo una vez.
            if self.modelo is None:

                if not RUTA_MODELO.is_file():
                    raise FileNotFoundError(
                        "No se encontró el modelo entrenado:\n\n"
                        f"{RUTA_MODELO}\n\n"
                        "Verifica que el archivo "
                        "'mobilenetv2_tomate.keras' esté "
                        "dentro de la carpeta 'modelos'."
                    )

                self.resultado.configure(
                    text="Cargando modelo..."
                )
                self.detalle.configure(text="")
                self.boton.configure(state="disabled")
                self.root.update_idletasks()

                try:
                    self.modelo = tf.keras.models.load_model(
                        RUTA_MODELO
                    )
                finally:
                    self.boton.configure(state="normal")

            # Ejecutar predicción.
            probabilidades = np.asarray(
                self.modelo.predict(lote, verbose=0)[0]
            )

            # Validar el número de categorías.
            if len(probabilidades) != len(CLASES):
                raise ValueError(
                    "El modelo no devuelve las tres categorías "
                    "esperadas. Comprueba el orden y las clases "
                    "utilizadas durante el entrenamiento."
                )

            if not np.all(np.isfinite(probabilidades)):
                raise ValueError(
                    "El modelo devolvió valores no válidos."
                )

            # Obtener la clase con mayor puntuación.
            indice = int(np.argmax(probabilidades))
            clase_predicha = CLASES[indice]
            nombre = NOMBRES[clase_predicha]

            confianza = float(probabilidades[indice]) * 100

            self.resultado.configure(
                text=(
                    f"Predicción: {nombre}\n"
                    f"Puntuación del modelo: {confianza:.2f}%"
                )
            )

            # Ordenar las categorías de mayor a menor puntuación.
            indices_ordenados = np.argsort(probabilidades)[::-1]

            lineas = ["Resultados por categoría:"]

            for indice_clase in indices_ordenados:
                clase = CLASES[indice_clase]
                valor = float(probabilidades[indice_clase]) * 100

                lineas.append(
                    f"• {NOMBRES[clase]}: {valor:.2f}%"
                )

            self.detalle.configure(
                text="\n".join(lineas)
            )

        except Exception as error:
            self.resultado.configure(
                text="No se pudo generar la predicción"
            )
            self.detalle.configure(text="")

            messagebox.showerror(
                "Error de predicción",
                "No fue posible generar la predicción.\n\n"
                f"Detalle: {error}"
            )


# =========================================================
# INICIAR APLICACIÓN
# =========================================================

if __name__ == "__main__":

    ventana = tk.Tk()
    app = AgroVisionApp(ventana)
    ventana.mainloop()
