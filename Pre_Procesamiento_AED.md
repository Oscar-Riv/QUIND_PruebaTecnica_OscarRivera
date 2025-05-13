Preprocesamiento y preparación de datos
Este archivo explica el flujo de preprocesamiento aplicado al dataset de Home Credit para dejarlo listo para modelos de clustering y predicción supervisada.
# 1. Carga de archivos
Se importaron los archivos .csv relevantes, incluyendo:
- application_train, application_test
- bureau, bureau_balance
- previous_application, installments_payments
- POS_CASH_balance, credit_card_balance

# 2. Limpieza de valores nulos
Se eliminaron columnas con más del 60% de valores faltantes.

# 3. Manejo de outliers
Se aplicó el método del rango intercuartil (IQR) para detectar y marcar outliers.
Para DAYS_EMPLOYED, se detectó un valor atípico estructural (365243) que fue reemplazado por NaN. Se encontraron otras variables con outliers, sin embargo, estas no se consideraron importantes para la limpieza de datos, ya que, seguramente no serían usadas posteriormente en los modelos por ser información no relevante, como por ejemplo los números de teléfonos de clientes o información no tan relacionada con riesgo crediticio.

# 4. Agregación de fuentes externas
Se agregaron datos de otras tablas (bureau, installments, previous_application, etc.) al dataset principal (application_train) usando el campo SK_ID_CURR.
Las variables numéricas se agruparon por cliente, aplicando funciones estadísticas (count, mean, max, min, sum), y se renombraron con un prefijo identificador (ej. BUREAU_, INST_, PREV_).

# 5. Análisis exploratorio inicial

Se generaron histogramas y boxplots para analizar:
- La distribución del monto del crédito (AMT_CREDIT)
- Ingresos (AMT_INCOME_TOTAL)
- Relación de estas variables con la columna objetivo (TARGET)
También se evaluó el desbalance de clases:
Solo ~8% de los clientes caen en mora (TARGET = 1).

# 6. Codificación de variables categóricas
- Variables binarias: LabelEncoder
- Variables con más de dos categorías: One-Hot Encoding
Luego se alinearon app_train y app_test para garantizar las mismas columnas.

# 7. Imputación final y guardado
Se imputaron valores faltantes restantes por mediana en ambos conjuntos (app_train, app_test).
Los datasets procesados se guardaron como:
- app_train_clean.csv
- app_test_clean.csv
