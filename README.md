AgroVision IA

Clasificación de enfermedades en hojas de tomate mediante redes neuronales

Proyecto académico individual para la asignatura Redes Neuronales, desarrollado en la Fundación Universitaria Católica del Norte.

Estudiante: Juan Pablo Tejada Márquez
Programa: Ingeniería en Analítica de Datos
Institución: Fundación Universitaria Católica del Norte

1. Descripción del proyecto

AgroVision IA es un prototipo de visión por computadora que utiliza modelos de aprendizaje profundo para clasificar imágenes de hojas de tomate en tres categorías:

Tizón temprano: Tomato___Early_blight

Tizón tardío: Tomato___Late_blight

Hoja sana: Tomato___healthy

El proyecto implementa y compara dos enfoques de redes neuronales: una red neuronal convolucional (CNN) y MobileNetV2 mediante transferencia de aprendizaje. También incorpora una interfaz gráfica para seleccionar una imagen y consultar la predicción del modelo.

2. Problema y objetivo

Las enfermedades de las plantas de tomate pueden afectar la productividad y la calidad de los cultivos. La identificación de síntomas mediante imágenes constituye una posible herramienta de apoyo para reconocer problemas fitosanitarios.

Objetivo general: desarrollar y evaluar dos modelos de redes neuronales para clasificar imágenes de hojas de tomate y comparar su desempeño mediante métricas calculadas sobre un conjunto de prueba.

3. Conjunto de datos

El experimento se realizó con un conjunto de 1.495 imágenes, distribuidas de la siguiente manera:

Subconjunto

Número de imágenes

Entrenamiento

1.046

Validación

224

Prueba

225

Total

1.495

La división se realizó utilizando la semilla aleatoria 42. El conjunto de prueba contiene 75 imágenes por categoría.

Las imágenes originales y los conjuntos divididos no se incluyen en este repositorio. Para reproducir el entrenamiento, es necesario preparar los datos, verificar las categorías y confirmar que las rutas coincidan con las esperadas por los scripts.

El archivo evidencias/resumen_dataset.json contiene información resumida sobre el conjunto de datos.

4. Modelos y resultados

Los modelos se evaluaron con las mismas 225 imágenes de prueba.

Métrica

CNN

MobileNetV2

Exactitud

71,11 %

89,33 %

Predicciones correctas

160 de 225

201 de 225

Predicciones incorrectas

65 de 225

24 de 225

F1 macro

69,09 %

89,26 %

En este experimento, MobileNetV2 obtuvo una exactitud superior en 18,22 puntos porcentuales y realizó 41 predicciones correctas más que la CNN.

En la identificación del tizón tardío, el recall fue del 40,00 % para la CNN y del 86,67 % para MobileNetV2. Esto indica que MobileNetV2 identificó una mayor proporción de los casos de esa categoría en el conjunto de prueba utilizado.

Estos resultados corresponden al experimento realizado y no garantizan el mismo desempeño con fotografías externas, condiciones ambientales diferentes u otras variedades de hojas.

5. Estructura del repositorio

La estructura principal del proyecto es la siguiente:

Redes-Neuronales-AgroVision_AI/
├── evidencias/
│   ├── CNN/
│   ├── MobileNetV2/
│   ├── comparacion_modelos/
│   ├── comparacion_modelos.csv
│   ├── comparacion_modelos.json
│   ├── informe_evidencias.md
│   ├── resumen_dataset.json
│   ├── prueba_interfaz_agrovision.png
│   └── prueba_interfaz_agrovision2.png
├── modelos/
│   ├── cnn_tomate.keras
│   └── mobilenetv2_tomate.keras
├── descargar_datos.py
├── dividir_datos.py
├── entrenar_mobilenet.py
├── entrenar_modelo.py
├── evaluar_cnn.py
├── generar_comparacion.py
├── generar_evidencias.py
├── generar_graficas.py
├── interfaz_agrovision.py
├── predecir_imagen.py
├── requirements.txt
├── verificar_datos.py
├── Informe_Final_Redes_Neuronales_AgroVision_AI_Juan_Pablo_Tejada.docx
└── README.md

Función de los principales archivos

Archivo

Función

descargar_datos.py

Descarga o preparación de los datos, según la configuración del script.

verificar_datos.py

Verificación de las categorías disponibles.

dividir_datos.py

División de los datos en entrenamiento, validación y prueba.

entrenar_modelo.py

Entrenamiento de la CNN.

entrenar_mobilenet.py

Entrenamiento de MobileNetV2.

evaluar_cnn.py

Evaluación del modelo CNN.

generar_comparacion.py

Comparación de los modelos.

generar_graficas.py

Generación de gráficas de resultados.

generar_evidencias.py

Generación u organización de evidencias.

predecir_imagen.py

Predicción sobre una imagen.

interfaz_agrovision.py

Interfaz gráfica de la aplicación.

requirements.txt

Dependencias de Python.

README.md

Documentación del proyecto.

Las carpetas evidencias/ y modelos/ contienen, respectivamente, los resultados experimentales publicados y los archivos de los modelos entrenados.

6. Tecnologías utilizadas

Python

TensorFlow y Keras

NumPy

pandas

scikit-learn

Matplotlib

Pillow

Tkinter

GitHub

7. Requisitos previos

Para ejecutar el proyecto se necesita:

Un computador con Windows y PowerShell, o un entorno compatible.

Una versión de Python compatible con las dependencias de requirements.txt.

Conexión a internet para instalar las dependencias y, si corresponde, preparar los datos.

Espacio disponible para los paquetes, las imágenes y los resultados del entrenamiento.

La ejecución de la interfaz requiere que el modelo entrenado correspondiente esté disponible en la ruta configurada en el código. Los archivos de los modelos están publicados en la carpeta modelos/ del repositorio, pero se debe verificar que las rutas utilizadas por la aplicación coincidan con esas ubicaciones.

8. Instalación y ejecución

8.1. Descargar el proyecto

Abrir el repositorio de AgroVision IA en GitHub.

Seleccionar Code → Download ZIP.

Extraer el archivo ZIP.

Abrir PowerShell en la carpeta raíz del proyecto, donde se encuentran interfaz_agrovision.py y requirements.txt.

8.2. Comprobar Python

Ejecutar los siguientes comandos por separado:

python --version
python -m pip --version

Si los comandos no funcionan, será necesario instalar Python y comprobar que esté disponible desde PowerShell.

8.3. Verificar los archivos

Ejecutar:

Test-Path .\interfaz_agrovision.py
Test-Path .\requirements.txt
Test-Path .\modelos\cnn_tomate.keras
Test-Path .\modelos\mobilenetv2_tomate.keras

Si la estructura está completa, los cuatro comandos deben devolver True.

8.4. Crear y activar un entorno virtual

Crear el entorno virtual:

python -m venv .venv

Activarlo:

.\.venv\Scripts\Activate.ps1

Si PowerShell bloquea la activación, se puede habilitar temporalmente para la sesión actual:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Después, ejecutar de nuevo el comando de activación. Este ajuste se limita a la sesión actual de PowerShell.

8.5. Instalar las dependencias

Con el entorno virtual activado, ejecutar cada comando por separado:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check

La instalación puede tardar varios minutos. Si aparecen errores, comprobar la versión de Python y los requisitos de las bibliotecas.

8.6. Ejecutar la interfaz

Desde la carpeta raíz del proyecto, ejecutar:

python .\interfaz_agrovision.py

Si la interfaz abre, pero no permite realizar predicciones, comprobar que el archivo del modelo requerido exista y que su ruta coincida con la configurada en el código.

9. Cómo probar una predicción

Cuando la interfaz y el modelo estén disponibles:

Abrir la aplicación.

Seleccionar una fotografía de una hoja de tomate en formato JPG, JPEG o PNG.

Esperar a que el modelo procese la imagen.

Revisar la categoría predicha.

Consultar las probabilidades mostradas por la aplicación, si están disponibles.

Repetir la prueba con otra imagen.

Las probabilidades corresponden a las salidas del modelo y no garantizan que una predicción sea correcta.

10. Entrenamiento y reproducibilidad

La ejecución de la interfaz y el entrenamiento son procesos diferentes.

Para probar la aplicación: se requieren las dependencias y el modelo entrenado en la ruta correspondiente.

Para repetir el entrenamiento: se deben preparar las imágenes, verificar su organización, ejecutar los scripts y evaluar los modelos sobre el conjunto de prueba.

La reproducción del experimento depende de conservar los datos, la configuración, las versiones de las bibliotecas y los procedimientos utilizados. Los resultados publicados documentan el experimento realizado, pero no garantizan que cualquier ejecución futura obtenga métricas idénticas.

11. Evidencias del proyecto

En la carpeta evidencias/ se publican archivos para documentar y revisar el desarrollo experimental, entre ellos:

Curvas de entrenamiento y validación.

Matrices de confusión.

Métricas de evaluación de cada modelo.

Historiales de entrenamiento.

Comparación de exactitud y F1 por categoría.

Capturas de la interfaz y pruebas de predicción.

Resúmenes y archivos de comparación de resultados.

Estas evidencias permiten revisar el desempeño obtenido por cada modelo sobre el conjunto de prueba.

12. Alcance y limitaciones

AgroVision IA es un prototipo académico experimental limitado a tres categorías de hojas de tomate.

El desempeño puede verse afectado por la iluminación, el fondo, el ángulo, la calidad de las imágenes y las diferencias entre los datos de entrenamiento y las fotografías externas.

La herramienta es de apoyo experimental y no sustituye el diagnóstico ni la evaluación de un especialista agrícola.

13. Autor

Juan Pablo Tejada Márquez

Proyecto académico individual para la asignatura Redes Neuronales.

Fundación Universitaria Católica del Norte.
