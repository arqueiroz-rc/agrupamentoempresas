import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('dados_empresas.csv')
print(df)

features = df[['Liquidez Corrente', 'Lucro Líquido', 'Valor de Mercado']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(scaled_features)

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_features)
df['Cluster'] = kmeans.labels_
print(df)

numeric_columns = ['Liquidez Corrente', 'Lucro Líquido', 'Valor de Mercado', 'Cluster']
clustered_data = df[numeric_columns].groupby('Cluster').mean()
print(clustered_data)

plt.figure(figsize=(10, 6))

for cluster in range(n_clusters):
    clustered_subset = df[df['Cluster'] == cluster]
    plt.scatter(clustered_subset['Liquidez Corrente'], 
                clustered_subset['Lucro Líquido'], 
                label=f'Cluster {cluster}', 
                s=100)  
    
plt.title('Agrupamento de Empresas por Liquidez Corrente e Lucro Líquido')
plt.xlabel('Liquidez Corrente')
plt.ylabel('Lucro Líquido')
plt.legend()
plt.grid(True)
plt.show()
