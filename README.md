# Segmentación y Clasificación de nódulos en Radiografías de Tórax

Este proyecto consiste en una interfaz que permite seleccionar un archivo local que contiene una radiografía de tórax para su análisis de nódulos. El objetivo principal es ejecutar una serie de algoritmos que realizan diferentes tareas de procesamiento de imágenes médicas y análisis de datos.

## Funcionalidades
**Selección de Archivo:** La interfaz permite al usuario seleccionar un archivo local que contiene la radiografía de tórax que se desea analizar.

**Segmentación de Pulmones:** Se utiliza un algoritmo de segmentación para identificar y resaltar los pulmones dentro de la radiografía. El resultado de esta segmentación se muestra en pantalla.

**Segmentación de Nódulo:** Se ejecuta un segundo algoritmo de segmentación para identificar y resaltar cualquier nódulo presente en la radiografía. El área del nódulo detectado se muestra en la interfaz.

**Análisis de Nódulo:** Se utiliza un tercer algoritmo para analizar el nódulo detectado y calcular probabilidades utilizando tres modelos de Machine Learning (ML). Estas probabilidades se utilizan para determinar la posibilidad de que el nódulo sea benigno o maligno. El resultado de este análisis se muestra en la sección "Diagnóstico" de la interfaz.

## Uso
1. Clona este repositorio en tu máquina local.
2. Abre la interfaz ejecutando el archivo interfaz.py.
3. Selecciona un archivo local que contenga una radiografía de tórax.
4. Ejecuta los algoritmos para visualizar los resultados en la interfaz.

## Requisitos del Sistema
* Python 3.x

* Bibliotecas de Python: numpy, opencv-python, scikit-learn, entre otras. Puedes instalarlas utilizando pip:
`pip install numpy opencv-python scikit-learn`
