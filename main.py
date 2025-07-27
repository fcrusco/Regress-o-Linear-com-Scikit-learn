# 1. Importar bibliotecas necessárias
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 2. Criar dados fictícios (Ex.: estudo de horas x nota na prova)
X = np.array([[1], [2], [3], [4], [5]])  # Horas de estudo
y = np.array([2, 4, 6, 8, 10])          # Nota obtida

# 3. Criar e treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# 4. Fazer uma previsão (Ex.: 6 horas de estudo)
nova_entrada = np.array([[6]])
previsao = modelo.predict(nova_entrada)

print(f"Se estudar 6 horas, a previsão de nota é: {previsao[0]:.2f}")

# 5. Visualizar os dados e a linha de regressão
plt.scatter(X, y, color="blue", label="Dados reais")
plt.plot(X, modelo.predict(X), color="red", label="Linha de regressão")
plt.xlabel("Horas de estudo")
plt.ylabel("Nota")
plt.title("Exemplo simples de Regressão Linear")
plt.legend()
plt.show()
