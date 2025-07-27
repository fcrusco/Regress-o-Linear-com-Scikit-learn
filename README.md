🐍 Machine Learning com Scikit-learn – Regressão Linear

<img width="795" height="679" alt="image" src="[https://github.com/user-attachments/assets/99ac67f4-a59d-4d72-bd9a-d559c77f3315](https://github.com/user-attachments/assets/99ac67f4-a59d-4d72-bd9a-d559c77f3315)" /\>


🐍 Machine Learning com Scikit-learn – Regressão Linear
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
Clone o repositório
Instale as dependências:
pip install -r requirements.txt
(Se não tiver o arquivo requirements.txt, instale manualmente):
pip install scikit-learn matplotlib numpy
python main.py

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
Regressão Linear: técnica estatística usada para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.
Machine Learning supervisionado: o modelo aprende a partir de exemplos com respostas conhecidas.

📚 Referências
Documentação do Scikit-learn - https://scikit-learn.org/stable/
Guia de Regressão Linear - https://scikit-learn.org/stable/modules/linear_model.html

-----------------------------------------------------------------------
