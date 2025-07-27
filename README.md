# 🐍 Machine Learning com Scikit-learn – Regressão Linear

<img width="795" height="679" alt="image" src="https://github.com/user-attachments/assets/d4c16f3a-06ea-4cdd-87bc-ce4177b4b539" />
<br>
<br>
Este projeto demonstra um exemplo simples de Machine Learning utilizando a biblioteca Scikit-learn, aplicando Regressão Linear para prever valores com base em dados numéricos.<br>
<br>
✅ Objetivo do Projeto<br>
Implementar um modelo de Regressão Linear para prever a relação entre duas variáveis (ex.: horas de estudo x nota obtida).<br>
Demonstrar o fluxo básico de Machine Learning supervisionado:<br>
Preparação dos dados<br>
Treinamento do modelo<br>
Predição<br>
Visualização dos resultados<br>
<br>
🛠️ Tecnologias Utilizadas<br>
Python 3.x<br>
Scikit-learn (para modelagem)<br>
NumPy (para manipulação numérica)<br>
Matplotlib (para visualização gráfica)<br>
<br>
▶️ Como Executar<br>
Clone o repositório<br>
Instale as dependências:<br>
pip install scikit-learn matplotlib numpy<br>
python main.py<br>
<br>
📊 Como o código funciona<br>
from sklearn.linear_model import LinearRegression<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
<br>
# Dados fictícios<br>
X = np.array([[1], [2], [3], [4], [5]])  # Horas de estudo<br>
y = np.array([2, 4, 6, 8, 10])          # Nota<br>
<br>
# Treinar o modelo<br>
modelo = LinearRegression()<br>
modelo.fit(X, y)<br>
<br>
# Prever nota para 6 horas de estudo<br>
previsao = modelo.predict([[6]])<br>
print(f"Se estudar 6 horas, previsão = {previsao[0]:.2f}")<br>
<br>
# Gráfico<br>
plt.scatter(X, y, color="blue")<br>
plt.plot(X, modelo.predict(X), color="red")<br>
plt.show()<br>
<br>
📷 Exemplo de Saída Gráfica<br>
O código gera um gráfico com os pontos reais (azuis) e a linha de regressão (vermelha), mostrando como o modelo ajustou os dados.<br>
<br>
📌 Conceitos Envolvidos<br>
Regressão Linear: técnica estatística usada para modelar a relação entre uma variável dependente e uma ou mais variáveis independentes.<br>
Machine Learning supervisionado: o modelo aprende a partir de exemplos com respostas conhecidas.<br>
<br>
📚 Referências<br>
Documentação do Scikit-learn - https://scikit-learn.org/stable/<br>
Guia de Regressão Linear - https://scikit-learn.org/stable/modules/linear_model.html<br>

-----------------------------------------------------------------------
