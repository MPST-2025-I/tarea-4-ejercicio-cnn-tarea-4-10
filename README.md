# Predicción de Precios de Viviendas y Demanda Eléctrica usando CNN

Este repositorio contiene la implementación de modelos de Redes Neuronales Convolucionales (CNN) para predecir el precio de las viviendas y la demanda eléctrica en Australia. El proyecto es parte de una tarea en la que se aplican técnicas de aprendizaje profundo para resolver problemas de regresión utilizando CNN.

---

## **Instrucciones de la Tarea**

### **1. Artículo Científico**
- Buscar un artículo científico donde se apliquen CNN.
- Hacer una discusión y mostrar los resultados en el notebook de solución de la tarea. (Sólo se solicita un artículo por equipo).

### **2. Implementación de CNN**
- **Dataset 1**: [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).
  - **Objetivo**: Predecir el precio de las viviendas usando CNN.
  - El dataset ya tiene todas las características necesarias, por lo que no se requiere preparación especial de los datos. Solo se debe buscar la mejor estructura de CNN que dé los mejores resultados.

- **Dataset 2**: [NSW Australia Electricity Demand](https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023/data).
  - **Objetivo**: Predecir el consumo de demanda eléctrica en Australia utilizando una serie temporal.
  - Aplicar las funciones de preparación de datos para CNN, como se vio en clase.

---

## **Lo que se hizo**

### **1. Preprocesamiento de Datos**
- Para el dataset de precios de viviendas:
  - Se normalizaron las características numéricas y se codificaron las categóricas.
  - Se dividieron los datos en conjuntos de entrenamiento y prueba.
- Para el dataset de demanda eléctrica:
  - Se extrajeron características temporales (hora, día de la semana, mes).
  - Se normalizaron los datos y se dividieron en secuencias para el entrenamiento de modelos de series temporales.

### **2. Modelos Implementados**
- **CNN Univariado**: Para predecir la demanda eléctrica utilizando una sola característica.
- **CNN Multivariado**: Para predecir la demanda eléctrica utilizando múltiples características.
- **CNN de Múltiples Cabeceras**: Para manejar múltiples entradas en el modelo.
- **CNN de Múltiples Salidas**: Para predecir múltiples objetivos simultáneamente.
- **CNN de Múltiples Pasos**: Para predecir varios pasos futuros en la serie temporal.

### **3. Evaluación de Modelos**
- Se entrenaron y evaluaron los modelos utilizando métricas como el error cuadrático medio (MSE) y el error absoluto medio (MAE).
- Se generaron gráficos de pérdida durante el entrenamiento para analizar el rendimiento de los modelos.

### **4. Resultados**
- Se obtuvieron resultados para cada modelo, comparando su rendimiento en términos de pérdida y precisión.
- Se discutieron los resultados en el notebook de solución.

---

## **Integrantes del Equipo**
- **Patricio Adulfo Villanueva Gio**
- **Melanie Michel Rodriguez**

---
