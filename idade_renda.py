# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregando o conjunto de dados
dados = pd.read_csv('dados_consolidados.csv')

# Codificando as características categóricas
# Convertendo 'Sexo' para valores numéricos
dados['Sexo'] = dados['Sexo'].map({'Masculino': 0, 'Feminino': 1})

# Convertendo 'Cor' para valores numéricos
mapa_cores = {
    'Indígena': 0, 'Branca': 2, 'Preta': 4,
    'Amarela': 6, 'Parda': 8, 'Sem declaração': 9
}
dados['Cor'] = dados['Cor'].map(mapa_cores)

# Selecionando as colunas de interesse
data_selected = dados[['Sexo', 'Idade', 'Renda']].values

# Aplicando k-means nos dados selecionados
# Aqui, considerei que a função 'kmeans' já foi definida anteriormente no seu código
labels, centroids = kmeans(data_selected, k=4)

# Retornando as contagens de cada label para verificar a distribuição dos clusters
cluster_counts = pd.Series(labels).value_counts()

# Calculando as medianas para Idade e Renda
idade_mediana = dados['Idade'].median()
renda_mediana = dados['Renda'].median()

# Definindo os quadrantes com base nas medianas
condicoes = [
    (dados['Idade'] > idade_mediana) & (dados['Renda'] > renda_mediana),
    (dados['Idade'] <= idade_mediana) & (dados['Renda'] > renda_mediana),
    (dados['Idade'] <= idade_mediana) & (dados['Renda'] <= renda_mediana)
]
quadrantes = ['Q1', 'Q2', 'Q3']
dados['Quadrante'] = np.select(condicoes, quadrantes, default='Q4')

# Plotando os dados por quadrante
plt.figure(figsize=(12, 7))
cores = ['blue', 'green', 'red', 'yellow']
for quadrante, cor in zip(quadrantes + ['Q4'], cores):
    subset = dados[dados['Quadrante'] == quadrante]
    plt.scatter(subset['Idade'], subset['Renda'], s=50, c=cor, label=quadrante, alpha=0.6)

# Adicionando linhas medianas e textos
plt.axvline(x=idade_mediana, color='black', linestyle='--')
plt.axhline(y=renda_mediana, color='black', linestyle='--')
plt.text(idade_mediana + 1, 0, f'Mediana Idade: {idade_mediana} anos', va='top', ha='right', color='black')
plt.text(0, renda_mediana + 100, f'Mediana Renda: R$ {renda_mediana}', va='top', ha='right', color='black')

# Configurando o título e os rótulos dos eixos
plt.title("Distribuição por Quadrantes (Idade x Renda)")
plt.xlabel("Idade")
plt.ylabel("Renda")
plt.legend()
plt.grid(True)
plt.show()

# Imprimindo as informações sobre os quadrantes
print(f"Mediana de Idade: {idade_mediana} anos")
print(f"Mediana de Renda: R$ {renda_mediana:,.2f}")
print("\nDescrição dos Quadrantes com base nas medianas de Idade e Renda:")
print("---------------------------------------------------------------")
print("Q1: Pessoas com Idade acima da mediana e Renda acima da mediana.")
print("Q2: Pessoas com Idade abaixo da mediana e Renda acima da mediana.")
print("Q3: Pessoas com Idade abaixo da mediana e Renda abaixo da mediana.")
print("Q4: Pessoas com Idade acima da mediana e Renda abaixo da mediana.")
