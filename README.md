🐍 Machine Learning com Scikit-learn – Regressão Linear\<br\>
\<br\>
\<img width="795" height="679" alt="image" src="[https://github.com/user-attachments/assets/99ac67f4-a59d-4d72-bd9a-d559c77f3315](https://github.com/user-attachments/assets/99ac67f4-a59d-4d72-bd9a-d559c77f3315)" /\>\<br\>
\<br\>
Este projeto demonstra um exemplo simples de Machine Learning utilizando a biblioteca Scikit-learn, aplicando Regressão Linear para prever valores com base em dados numéricos.\<br\>
\<br\>
✅ Objetivo do Projeto\<br\>
Implementar um modelo de Regressão Linear para prever a relação entre duas variáveis (ex.: horas de estudo x nota obtida).\<br\>
\<br\>
Demonstrar o fluxo básico de Machine Learning supervisionado:\<br\>
Preparação dos dados\<br\>
Treinamento do modelo\<br\>
Predição\<br\>
Visualização dos resultados\<br\>
\<br\>
🛠️ Tecnologias Utilizadas\<br\>
Python 3.x\<br\>
Scikit-learn (para modelagem)\<br\>
NumPy (para manipulação numérica)\<br\>
Matplotlib (para visualização gráfica)\<br\>
\<br\>
▶️ Como Executar\<br\>
Clone o repositório:\<br\>
Instale as dependências:\<br\>
pip install scikit-learn matplotlib numpy\<br\>
\<br\>
📊 Como o código funciona\<br\>
from sklearn.linear\_model import LinearRegression\<br\>
import numpy as np\<br\>
import matplotlib.pyplot as plt\<br\>
\<br\>

# Dados fictícios\<br\>

X = np.array([[1], [2], [3], [4], [5]])  \# Horas de estudo\<br\>
y = np.array([2, 4, 6, 8, 10])           \# Nota\<br\>
\<br\>

# Treinar o modelo\<br\>

modelo = LinearRegression()\<br\>
modelo.fit(X, y)\<br\>
\<br\>

# Prever nota para 6 horas de estudo\<br\>

previsao = modelo.predict([[6]])\<br\>
print(f"Se estudar 6 horas, previsão = {previsao[0]:.2f}")\<br\>
\<br\>

# Gráfico\<br\>

plt.scatter(X, y, color="blue")\<br\>
plt.plot(X, modelo.predict(X), color="red")\<br\>
plt.show()\<br\>
\<br\>
📷 Exemplo de Saída Gráfica\<br\>
O código gera um gráfico com os pontos reais (azuis) e a linha de regressão (vermelha), mostrando como o modelo ajustou os dados.\<br\>
\<br\>
📌 Conceitos Envolvidos\<br\>
Regressão Linear: Técnica estatística usada para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.\<br\>
\<br\>
Machine Learning supervisionado: O modelo aprende a partir de exemplos com respostas conhecidas.\<br\>
\<br\>
📚 Referências\<br\>
Documentação do Scikit-learn - [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)\<br\>
Guia de Regressão Linear - [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)\<br\>
\<br\>
\---\<br\>
\<br\>
🐍 Machine Learning with Scikit-learn – Linear Regression\<br\>
\<br\>
This project demonstrates a simple example of Machine Learning using the Scikit-learn library, applying Linear Regression to predict values based on numerical data.\<br\>
\<br\>
✅ Project Objective\<br\>
Implement a Linear Regression model to predict the relationship between two variables (e.g., study hours x grade obtained).\<br\>
\<br\>
Demonstrate the basic supervised Machine Learning workflow:\<br\>
Data preparation\<br\>
Model training\<br\>
Prediction\<br\>
Result visualization\<br\>
\<br\>
🛠️ Technologies Used\<br\>
Python 3.x\<br\>
Scikit-learn (for modeling)\<br\>
NumPy (for numerical manipulation)\<br\>
Matplotlib (for graphical visualization)\<br\>
\<br\>
▶️ How to Run\<br\>
Clone the repository:\<br\>
Install dependencies:\<br\>
pip install scikit-learn matplotlib numpy\<br\>
\<br\>
📊 How the code works\<br\>
from sklearn.linear\_model import LinearRegression\<br\>
import numpy as np\<br\>
import matplotlib.pyplot as plt\<br\>
\<br\>
Fictitious data\<br\>
X = np.array([[1], [2], [3], [4], [5]]) \# Study hours\<br\>
y = np.array([2, 4, 6, 8, 10]) \# Grade\<br\>
\<br\>
Train the model\<br\>
modelo = LinearRegression()\<br\>
modelo.fit(X, y)\<br\>
\<br\>
Predict grade for 6 hours of study\<br\>
previsao = modelo.predict([[6]])\<br\>
print(f"If you study 6 hours, prediction = {previsao[0]:.2f}")\<br\>
\<br\>
Graph\<br\>
plt.scatter(X, y, color="blue")\<br\>
plt.plot(X, modelo.predict(X), color="red")\<br\>
plt.show()\<br\>
\<br\>
📷 Graphical Output Example\<br\>
The code generates a graph with the real points (blue) and the regression line (red), showing how the model adjusted the data.\<br\>
\<br\>
📌 Concepts Involved\<br\>
Linear Regression: A statistical technique used to model the relationship between a dependent variable and one or more independent variables.\<br\>
\<br\>
Supervised Machine Learning: The model learns from examples with known answers.\<br\>
\<br\>
📚 References\<br\>
Scikit-learn Documentation - [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)\<br\>
Linear Regression Guide - [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)\<br\>
\<br\>
\---\<br\>
\<br\>
🐍 Machine Learning con Scikit-learn – Regresión Lineal\<br\>
\<br\>
Este proyecto demuestra un ejemplo simple de Machine Learning utilizando la biblioteca Scikit-learn, aplicando Regresión Lineal para predecir valores con base en datos numéricos.\<br\>
\<br\>
✅ Objetivo del Proyecto\<br\>
Implementar un modelo de Regresión Lineal para predecir la relación entre dos variables (ej.: horas de estudio x calificación obtenida).\<br\>
\<br\>
Demostrar el flujo básico de Machine Learning supervisado:\<br\>
Preparación de los datos\<br\>
Entrenamiento del modelo\<br\>
Predicción\<br\>
Visualización de los resultados\<br\>
\<br\>
🛠️ Tecnologías Utilizadas\<br\>
Python 3.x\<br\>
Scikit-learn (para modelado)\<br\>
NumPy (para manipulación numérica)\<br\>
Matplotlib (para visualización gráfica)\<br\>
\<br\>
▶️ Cómo Ejecutar\<br\>
Clona el repositorio:\<br\>
Instala las dependencias:\<br\>
pip install scikit-learn matplotlib numpy\<br\>
\<br\>
📊 Cómo funciona el código\<br\>
from sklearn.linear\_model import LinearRegression\<br\>
import numpy as np\<br\>
import matplotlib.pyplot as plt\<br\>
\<br\>
Datos ficticios\<br\>
X = np.array([[1], [2], [3], [4], [5]])  \# Horas de estudio\<br\>
y = np.array([2, 4, 6, 8, 10])           \# Calificación\<br\>
\<br\>
Entrenar el modelo\<br\>
modelo = LinearRegression()\<br\>
modelo.fit(X, y)\<br\>
\<br\>
Predecir calificación para 6 horas de estudio\<br\>
previsao = modelo.predict([[6]])\<br\>
print(f"Si estudias 6 horas, la predicción = {previsao[0]:.2f}")\<br\>
\<br\>
Gráfico\<br\>
plt.scatter(X, y, color="blue")\<br\>
plt.plot(X, modelo.predict(X), color="red")\<br\>
plt.show()\<br\>
\<br\>
📷 Ejemplo de Salida Gráfica\<br\>
El código genera un gráfico con los puntos reales (azules) y la línea de regresión (roja), mostrando cómo el modelo ajustó los datos.\<br\>
\<br\>
📌 Conceitos Envolvidos\<br\>
Regresión Lineal: Técnica estadística usada para modelar la relación entre una variable dependiente y una o más variables independientes.\<br\>
\<br\>
Machine Learning supervisado: El modelo aprende a partir de ejemplos con respuestas conocidas.\<br\>
\<br\>
📚 Referencias\<br\>
Documentación de Scikit-learn - [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)\<br\>
Guía de Regresión Lineal - [https://scikit-learn.org/stable/modules/linear\_model.html](https://scikit-learn.org/stable/modules/linear_model.html)\<br\>
