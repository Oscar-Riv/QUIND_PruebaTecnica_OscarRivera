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
### 🔹 3. Modelos comparativos iniciales

Antes de aplicar LightGBM, se evaluaron dos modelos de clasificación clásicos: **Regresión Logística** y **Random Forest**. Ambos se entrenaron con el conjunto balanceado y las 20 variables seleccionadas previamente.

#### 📈 Resultados de Regresión Logística

- Buen desempeño identificando la clase minoritaria (clientes morosos).
- Recall de 0.67 en la clase 1, aunque con precisión limitada.
- AUC competitivo, superior a 0.73.

```
Logistic Regression
              precision    recall  f1-score   support

         0.0       0.96      0.68      0.80     70672
         1.0       0.16      0.67      0.25      6206

    accuracy                           0.68     76878
   macro avg       0.56      0.68      0.53     76878
weighted avg       0.89      0.68      0.75     76878

AUC: 0.7377
```

#### 🌲 Resultados de Random Forest

- Alta precisión en clase 0, pero recall muy bajo en la clase 1.
- El modelo prácticamente no identifica clientes morosos.
- AUC inferior al de regresión logística.

```
Random Forest
              precision    recall  f1-score   support

         0.0       0.92      1.00      0.96     70672
         1.0       0.57      0.00      0.01      6206

    accuracy                           0.92     76878
   macro avg       0.74      0.50      0.48     76878
weighted avg       0.89      0.92      0.88     76878

AUC: 0.7199
```

### ✅ Conclusión preliminar

Aunque ambos modelos fueron útiles como referencia, se optó por avanzar con **LightGBM** debido a que logró un mejor equilibrio entre precisión y recall, además de mantener un AUC superior. En la siguiente sección se describe su configuración y optimización.

### 🔹 4. Modelado con LightGBM

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

### 🔹 5. Métricas de Evaluación

Se utilizaron las siguientes métricas sobre el conjunto de prueba para evaluar el desempeño del modelo:

- **Precision, Recall y F1-score** de ambas clases (`0`: no moroso, `1`: moroso)
- **AUC (Área bajo la curva ROC)** como métrica principal para evaluar capacidad discriminativa

#### 📊 Resultados obtenidos con LightGBM

El modelo logró mejorar respecto a los modelos anteriores, en especial en la clase minoritaria (`1`), con un recall elevado y AUC competitivo:

```
LightGBM
              precision    recall  f1-score   support

         0.0       0.96      0.68      0.80     70672
         1.0       0.16      0.68      0.26      6206

    accuracy                           0.68     76878
   macro avg       0.56      0.68      0.53     76878
weighted avg       0.90      0.68      0.75     76878

AUC: 0.7465
```

El modelo mostró un balance razonable al identificar clientes con riesgo de mora, mejorando la capacidad de detección sin sacrificar excesivamente la precisión.

---

### 🔹 6. Interpretación de resultados

El modelo LightGBM optimizado logró resultados **comparables y superiores** a los modelos base (Regresión Logística y Random Forest), especialmente en la **detección de morosos** (clase 1), lo cual es crucial en un sistema de gestión de riesgo financiero.

#### 📌 Evaluación de efectividad

- **Recall (0.68) en la clase 1** indica que el modelo identifica aproximadamente el **68% de los clientes que efectivamente caerán en mora**. Esto es una mejora sustancial frente a Random Forest (recall ~0).
- **Precisión (0.16)** Una precisión del 16% en la clase de morosos implica que, aunque el modelo logra identificar a la mayoría de los clientes que realmente caerán en mora (recall alto), también genera una cantidad considerable de falsos positivos. En un contexto real, esto puede traducirse en decisiones conservadoras: limitar o condicionar el crédito a clientes que en realidad habrían pagado. No obstante, esta estrategia puede ser útil para prevenir pérdidas si se implementa como un sistema de alerta temprana, especialmente si se complementa con análisis adicionales o ajustes en el umbral de decisión.
- **AUC (0.746)** confirma que el modelo tiene una **buena capacidad de discriminación global**, distinguiendo entre clientes en riesgo y no en riesgo con un área bajo la curva ROC del 74.6%.

En resumen, el modelo es **exitoso en su propósito principal: detectar posibles morosos con buena cobertura**, aunque sacrificando algo de precisión debido al desbalance de clases. Para un sistema real, este comportamiento es aceptable si el objetivo es **minimizar el riesgo crediticio a través de medidas preventivas**.

---

### 📊 Importancia de las variables

El gráfico siguiente muestra las 20 variables más influyentes en el modelo LightGBM, basadas en su ganancia acumulada en la construcción de árboles:

![Importancia de variables - LightGBM](images/importance_lbgm.png)

#### 🧠 Análisis de variables clave

- Las variables más importantes provienen de:
  - **Scores externos** (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`)
  - **Edad (`DAYS_BIRTH`)** y **antigüedad laboral (`DAYS_EMPLOYED`)**
  - **Historial de crédito en buró** (`BUREAU_*`)
- Esto confirma que el **riesgo crediticio se explica mejor por fuentes externas y comportamientos históricos**, más que por variables socioeconómicas superficiales.

Además, variables como `CODE_GENDER_F`, `EDUCATION_TYPE` y `REGION_RATING_CLIENT` también aparecen en el top, indicando que **ciertas características demográficas tienen un impacto no despreciable**.

El modelo aprovecha estas variables para construir árboles de decisión eficientes que diferencian perfiles de bajo y alto riesgo.

---
### 🔹 7. Conclusión

El modelo final puede ser usado como base para priorizar evaluación de riesgo, con posibilidad de ser mejorado vía:

- Ajuste de umbrales
- Técnicas de remuestreo (SMOTE)
- Modelos más complejos (XGBoost)
