# === 1. Carga y configuración ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
import gc

# Cargar dataset procesado, solo train ya que se va a desarrollar un modelo no supervisado
app_train = pd.read_csv("app_train_clean.csv")

# === 2. Limpieza de variables ===
# Seleccion de columnas numéricas, las booleanas no suelen ser usadas para modelos de segmentación
numeric = app_train.select_dtypes(include=['number'])

# Eliminar columnas con muy baja varianza
vt = VarianceThreshold(threshold=0.01)
vt.fit(numeric)
numeric_reduced = numeric.loc[:, vt.get_support()]

# Eliminar variables con alta correlación (> 0.9)
corr_matrix = numeric_reduced.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
numeric_filtered = numeric_reduced.drop(columns=to_drop)

# === 3. Selección de variables por varianza intercluster con KMeans ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_filtered)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_filtered.columns)
feature_variance = centroids.var().sort_values(ascending=False)

# Selección final de variables
important_vars = feature_variance.head(10).index.tolist()
extra_vars = [
    'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
]
features_clust = important_vars + extra_vars

# === 4. Determinación de número óptimo de clusters (método del codo) ===
X = app_train[features_clust].copy()
X_scaled = scaler.fit_transform(X)

inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# === 5. Segmentación final con KMeans ===
k_opt = 7  # Elegido visualmente
kmeans = KMeans(n_clusters=k_opt, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
app_train['CLUSTER'] = clusters

# === 6. Análisis por cluster ===
# Estadistica descriptiva de los clusters
cluster_summary = app_train.groupby('CLUSTER')[features_clust + ['TARGET']].mean().round(2)
cluster_summary.to_csv("cluster_sum.csv", sep=',', index=False)

# Tasa de mora por cluster
risk_by_cluster = app_train.groupby('CLUSTER')['TARGET'].mean()
print("Tasa de mora por cluster:")
print(risk_by_cluster)

# Distribución de clientes por cluster
cluster_sizes = app_train['CLUSTER'].value_counts().sort_index()
print("Clientes por cluster:")
print(cluster_sizes)

# === 7. Visualización ===
variables_plot = [
    'AMT_CREDIT', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL',
    'EXT_SOURCE_1', 'INST_AMT_INSTALMENT_SUM'
]

sns.set(style="whitegrid")
for var in variables_plot:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='CLUSTER', y=var, data=app_train)
    plt.title(f'Distribución de {var} por CLUSTER')
    plt.xlabel('CLUSTER')
    plt.ylabel(var)
    plt.tight_layout()
    plt.show()

# Ratio crédito/ingreso por cluster
app_train['CREDIT_INCOME_RATIO'] = app_train['AMT_CREDIT'] / app_train['AMT_INCOME_TOTAL']
sns.boxplot(x='CLUSTER', y='CREDIT_INCOME_RATIO', data=app_train)
plt.title('Ratio crédito/ingreso por cluster')
plt.show()

# Tasa de mora promedio por cluster
sns.barplot(x='CLUSTER', y='TARGET', data=app_train, estimator='mean')
plt.title('Tasa de mora promedio por cluster')
plt.show()
