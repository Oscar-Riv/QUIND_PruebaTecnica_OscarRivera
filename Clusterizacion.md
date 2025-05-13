## Análisis de Segmentación con Clustering (Modelo No Supervisado)

Este análisis tiene como objetivo segmentar a los clientes según características numéricas para identificar perfiles de riesgo diferenciados.

---

### 🔹 1. Selección de variables

Se aplicaron tres pasos para reducir el número de variables:

1. **Filtro de baja varianza** (`threshold=0.01`)
2. **Eliminación de variables altamente correlacionadas** (`r > 0.9`)
3. **Ranking por varianza inter-cluster (KMeans)** → se eligieron las 10 más informativas

A estas se sumaron variables clave como `AMT_CREDIT`, `DAYS_BIRTH`, `EXT_SOURCE_1/2/3`.

---

### 🔹 2. Determinación del número de clusters

Se utilizó el **método del codo** (inercia) para determinar el número óptimo de clusters (`k=7`), equilibrando simplicidad y segmentación significativa.

---

### 🔹 3. Ejecución de KMeans

Se aplicó `KMeans(n_clusters=7)` sobre los datos estandarizados.  
A cada cliente se le asignó un `CLUSTER` entre 0 y 6.

---

### 🔹 4. Análisis de resultados

Se analizó:

- **Promedios por cluster** (edad, crédito, ingreso)
- **Distribución de clientes por cluster**
- **Tasa de mora (`TARGET`) por cluster**

Los clusters mostraron diferencias notables:
- Algunos concentraron alta mora con menor ingreso y edad más joven
- Otros presentaron perfiles estables y baja mora

---

### 🔹 5. Visualizaciones generadas

- **Boxplots** de variables financieras por cluster
- **Ratio crédito/ingreso** por cluster
- **Tasa de mora promedio** por cluster

Estas gráficas permitieron interpretar los perfiles y diferencias estructurales entre segmentos.

