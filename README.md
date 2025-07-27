🐍 Machine Learning com Scikit-learn – Regressão Linear

<img width="795" height="679" alt="image" src="https://github.com/user-attachments/assets/99ac67f4-a59d-4d72-bd9a-d559c77f3315" />

Este projeto demonstra um exemplo simples de Machine Learning utilizando a biblioteca Scikit-learn, aplicando Regressão Linear para prever valores com base em dados numéricos.

✅ Objetivo do Projeto
Implementar um modelo de Regressão Linear para prever a relação entre duas variáveis (ex.: horas de estudo x nota obtida).

Demonstrar o fluxo básico de Machine Learning supervisionado:
Preparação dos dados
Treinamento do modelo
Predição
Visualização dos resultados

🛠️ Tecnologias Utilizadas
Python 3.x
Scikit-learn (para modelagem)
NumPy (para manipulação numérica)
Matplotlib (para visualização gráfica)

▶️ Como Executar
Clone o repositório:
Instale as dependências:
pip install scikit-learn matplotlib numpy

📊 Como o código funciona
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Dados fictícios
X = np.array([[1], [2], [3], [4], [5]])  # Horas de estudo
y = np.array([2, 4, 6, 8, 10])          # Nota

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Prever nota para 6 horas de estudo
previsao = modelo.predict([[6]])
print(f"Se estudar 6 horas, previsão = {previsao[0]:.2f}")

# Gráfico
plt.scatter(X, y, color="blue")
plt.plot(X, modelo.predict(X), color="red")
plt.show()

📷 Exemplo de Saída Gráfica
O código gera um gráfico com os pontos reais (azuis) e a linha de regressão (vermelha), mostrando como o modelo ajustou os dados.

📌 Conceitos Envolvidos
Regressão Linear: Técnica estatística usada para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.

Machine Learning supervisionado: O modelo aprende a partir de exemplos com respostas conhecidas.

📚 Referências
Documentação do Scikit-learn - https://scikit-learn.org/stable/
Guia de Regressão Linear - https://scikit-learn.org/stable/modules/linear_model.html

-----------------------------------------------

🐍 Machine Learning with Scikit-learn – Linear Regression

This project demonstrates a simple example of Machine Learning using the Scikit-learn library, applying Linear Regression to predict values based on numerical data.

✅ Project Objective
Implement a Linear Regression model to predict the relationship between two variables (e.g., study hours x grade obtained).

Demonstrate the basic supervised Machine Learning workflow:
Data preparation
Model training
Prediction
Result visualization

🛠️ Technologies Used
Python 3.x
Scikit-learn (for modeling)
NumPy (for numerical manipulation)
Matplotlib (for graphical visualization)

▶️ How to Run
Clone the repository:
Install dependencies:
pip install scikit-learn matplotlib numpy

📊 How the code works
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

Fictitious data
X = np.array([[1], [2], [3], [4], [5]]) # Study hours
y = np.array([2, 4, 6, 8, 10]) # Grade

Train the model
modelo = LinearRegression()
modelo.fit(X, y)

Predict grade for 6 hours of study
previsao = modelo.predict([[6]])
print(f"If you study 6 hours, prediction = {previsao[0]:.2f}")

Graph
plt.scatter(X, y, color="blue")
plt.plot(X, modelo.predict(X), color="red")
plt.show()

📷 Graphical Output Example
The code generates a graph with the real points (blue) and the regression line (red), showing how the model adjusted the data.

📌 Concepts Involved
Linear Regression: A statistical technique used to model the relationship between a dependent variable and one or more independent variables.

Supervised Machine Learning: The model learns from examples with known answers.

📚 References
Scikit-learn Documentation - https://scikit-learn.org/stable/
Linear Regression Guide - https://scikit-learn.org/stable/modules/linear_model.html

-----------------------------------------------------

🐍 Machine Learning con Scikit-learn – Regresión Lineal

Este proyecto demuestra un ejemplo simple de Machine Learning utilizando la biblioteca Scikit-learn, aplicando Regresión Lineal para predecir valores con base en datos numéricos.

✅ Objetivo del Proyecto
Implementar un modelo de Regresión Lineal para predecir la relación entre dos variables (ej.: horas de estudio x calificación obtenida).

Demostrar el flujo básico de Machine Learning supervisado:
Preparación de los datos
Entrenamiento del modelo
Predicción
Visualización de los resultados

🛠️ Tecnologías Utilizadas
Python 3.x
Scikit-learn (para modelado)
NumPy (para manipulación numérica)
Matplotlib (para visualización gráfica)

▶️ Cómo Ejecutar
Clona el repositorio:
Instala las dependencias:
pip install scikit-learn matplotlib numpy

📊 Cómo funciona el código
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

Datos ficticios
X = np.array([[1], [2], [3], [4], [5]])  # Horas de estudio
y = np.array([2, 4, 6, 8, 10])           # Calificación

Entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

Predecir calificación para 6 horas de estudio
previsao = modelo.predict([[6]])
print(f"Si estudias 6 horas, la predicción = {previsao[0]:.2f}")

Gráfico
plt.scatter(X, y, color="blue")
plt.plot(X, modelo.predict(X), color="red")
plt.show()

📷 Ejemplo de Salida Gráfica
El código genera un gráfico con los puntos reales (azules) y la línea de regresión (roja), mostrando cómo el modelo ajustó los datos.

📌 Conceptos Involucrados
Regresión Lineal: Técnica estadística usada para modelar la relación entre una variable dependiente y una o más variables independientes.

Machine Learning supervisado: El modelo aprende a partir de ejemplos con respuestas conocidas.

📚 Referencias
Documentación de Scikit-learn - https://scikit-learn.org/stable/
Guía de Regresión Lineal - https://scikit-learn.org/stable/modules/linear_model.html
