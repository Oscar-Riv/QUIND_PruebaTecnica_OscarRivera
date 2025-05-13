## Predicción de Mora (Modelo Supervisado)

Este análisis tiene como objetivo predecir la probabilidad de que un cliente incurra en mora utilizando algoritmos de clasificación.

---

### 🔹 1. Preprocesamiento de variables

Se aplicaron filtros previos para reducir dimensionalidad:

- **Eliminación de baja varianza** (`threshold=0.01`)
- **Eliminación de variables altamente correlacionadas** (`r > 0.9`)
- **Selección de las 20 mejores variables** usando `SelectKBest` con estadístico ANOVA F (`f_classif`)

---

### 🔹 2. Separación de datos

El conjunto de datos fue separado en:

- `X`: variables predictoras
- `y`: variable objetivo (`TARGET`)
- División en `train` y `test` (75/25, estratificada)

Se aplicó `StandardScaler` para estandarizar los datos antes del entrenamiento.

---

### 🔹 3. Modelado con LightGBM

Se utilizó el clasificador `LGBMClassifier`, configurado para manejar desbalance con:

- `scale_pos_weight = ratio_entre_clases`

Se realizó ajuste de hiperparámetros usando `RandomizedSearchCV` con validación cruzada (CV=3) y 30 combinaciones aleatorias.

Parámetros optimizados:
- `num_leaves`
- `learning_rate`
- `max_depth`
- `n_estimators`
- `min_child_samples`
- `subsample`
- `colsample_bytree`

---

### 🔹 4. Métricas de Evaluación

Se utilizaron las siguientes métricas sobre el conjunto de prueba:

- **Precision, Recall y F1-score** de ambas clases (moroso / no moroso)
- **AUC (Área bajo la curva ROC)** para evaluar capacidad discriminativa

---

### 🔹 5. Interpretación de resultados

- El modelo LightGBM optimizado obtuvo mejor desempeño que la regresión logística y random forest en AUC y F1-score.
- Se identificaron las variables más importantes para la predicción (visualizadas con un gráfico de barras).
- A pesar de no alcanzar precisión perfecta, el modelo logra un **buen balance entre identificar morosos y minimizar falsos positivos**.

---

### 🔹 6. Conclusión

El modelo final puede ser usado como base para priorizar evaluación de riesgo, con posibilidad de ser mejorado vía:

- Ajuste de umbrales
- Técnicas de remuestreo (SMOTE)
- Modelos más complejos (XGBoost)
