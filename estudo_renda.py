import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Leitura dos dados a partir de um arquivo CSV (substitua 'seu_arquivo.csv' pelo nome do seu arquivo)
dados = pd.read_csv('seu_arquivo.csv')

# Preencher valores NaN em 'Anos de Estudo' com a mediana
dados['Anos de Estudo'].fillna(dados['Anos de Estudo'].median(), inplace=True)

# Verificar o tipo de dados da coluna 'Anos de Estudo'
if dados['Anos de Estudo'].dtype != 'float64':
    # Remover o texto 'anos' e converter a coluna para numérica
    dados['Anos de Estudo'] = dados['Anos de Estudo'].str.replace(' anos', '').astype(float)

# Padronizar os dados após a conversão da coluna para numérica
scaler = StandardScaler()
data_kmeans_study_income = dados[['Anos de Estudo', 'Renda']]
data_scaled_study_income = scaler.fit_transform(data_kmeans_study_income)

# Continuar com o clustering K-means
k = 4  # Definir o número de clusters
kmeans_study_income = KMeans(n_clusters=k, random_state=0).fit(data_scaled_study_income)
dados['Study_Income_Cluster'] = kmeans_study_income.labels_
cluster_distribution_study_income = dados['Study_Income_Cluster'].value_counts()

# Selecionar novamente as colunas de interesse
data_kmeans_study_income = dados[['Anos de Estudo', 'Renda']]

# Padronizar os dados
data_scaled_study_income = scaler.fit_transform(data_kmeans_study_income)

# Aplicar o K-means novamente
kmeans_study_income = KMeans(n_clusters=k, random_state=0).fit(data_scaled_study_income)

# Adicionar os rótulos dos clusters aos dados
dados['Study_Income_Cluster'] = kmeans_study_income.labels_

# Verificar a distribuição dos clusters
cluster_distribution_study_income = dados['Study_Income_Cluster'].value_counts()

# Ordenar os clusters com base nos valores médios de Anos de Estudo e Renda
sorted_clusters = cluster_characteristics.sort_values(by=['Anos de Estudo', 'Renda']).index

# Mapear cores para os clusters ordenados
color_mapping = {sorted_clusters[i]: color for i, color in enumerate(['red', 'orange', 'blue', 'green'])}

plt.figure(figsize=(14, 8))

# Gráfico de dispersão para os clusters
for cluster in sorted_clusters:
    subset = dados[dados['Study_Income_Cluster'] == cluster]
    plt.scatter(subset['Anos de Estudo'], subset['Renda'], s=50, c=color_mapping[cluster], label=f'Cluster {cluster}', alpha=0.6)

# Adicionar linhas e textos das medianas
plt.axvline(x=study_median, color='black', linestyle='--')
plt.axhline(y=income_median, color='black', linestyle='--')
plt.text(study_median + 1, 0, f'Mediana Anos de Estudo: {study_median} anos', verticalalignment='top', horizontalalignment='right', color='black')
plt.text(0, income_median + 100, f'Mediana Renda: R$ {income_median:,.2f}', verticalalignment='top', horizontalalignment='right', color='black')

plt.title("Distribuição por Cluster (Anos de Estudo x Renda)")
plt.xlabel("Anos de Estudo")
plt.ylabel("Renda")
plt.legend()
plt.grid(True)
plt.show()

# Calcular os valores médios para cada cluster para entender suas características
cluster_characteristics = dados.groupby('Study_Income_Cluster')[['Anos de Estudo', 'Renda']].mean().sort_values(by='Renda', ascending=False)
cluster_characteristics

# Calcular as medianas de Anos de Estudo e Renda
study_median = dados['Anos de Estudo'].median()
income_median = dados['Renda'].median()

# Imprimir as medianas de Anos de Estudo e Renda
print(f"Mediana de Anos de Estudo: {study_median} anos")
print(f"Mediana de Renda: R$ {income_median:,.2f}")

# Imprimir a descrição dos clusters
print("Descrição dos Clusters com base nas medianas de Anos de Estudo e Renda:")
print("-------------------------------------------------------------------------")
print("Cluster 0: Pessoas com Anos de Estudo próximos da mediana e Renda abaixo da mediana.")
print("Cluster 1: Pessoas com Anos de Estudo bem abaixo da mediana e Renda bem abaixo da mediana.")
print("Cluster 2: Pessoas com Anos de Estudo acima da mediana e Renda muito acima da mediana.")
print("Cluster 3: Pessoas com Anos de Estudo acima da mediana e Renda acima da mediana.")

# Filtrar os dados para o Cluster 2
cluster_2_data = dados[dados['Study_Income_Cluster'] == 2]

# Estatísticas descritivas para Anos de Estudo e Renda no Cluster 2
cluster_2_stats = cluster_2_data[['Anos de Estudo', 'Renda']].describe()
cluster_2_stats

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# Histograma de Anos de Estudo no Cluster 2
axes[0, 0].hist(cluster_2_data['Anos de Estudo'], bins=15, color='cyan', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Histograma dos Anos de Estudo (Cluster 2)')
axes[0, 0].set_xlabel('Anos de Estudo')
axes[0, 0].set_ylabel('Contagem')
axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Histograma de Renda no Cluster 2
axes[0, 1].hist(cluster_2_data['Renda'], bins=15, color='magenta', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Histograma da Renda (Cluster 2)')
axes[0, 1].set_xlabel('Renda')
axes[0, 1].set_ylabel('Contagem')
axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# Gráfico de dispersão de Anos de Estudo vs Renda no Cluster 2
axes[1, 0].scatter(cluster_2_data['Anos de Estudo'], cluster_2_data['Renda'], color='limegreen', alpha=0.6)
axes[1, 0].set_title('Gráfico de Dispersão dos Anos de Estudo x Renda (Cluster 2)')
axes[1, 0].set_xlabel('Anos de Estudo')
axes[1, 0].set_ylabel('Renda')
axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Remover o subplot não utilizado
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.show()
