# === 1. Carga de Datos ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# Cargar archivos CSV principales
application_train = pd.read_csv('application_train.csv')
application_test = pd.read_csv('application_test.csv')

bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')
previous_application = pd.read_csv('previous_application.csv')
installments_payments = pd.read_csv('installments_payments.csv')
poscash_balance = pd.read_csv('POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('credit_card_balance.csv')

# === 2. Limpieza Inicial ===
# Marcar outliers extremos conocidos
application_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
application_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# Eliminar columnas con más del 60% de valores faltantes
def drop_high_nan(df, threshold=0.6):
    nan_percent = df.isnull().mean()
    drop_cols = nan_percent[nan_percent > threshold].index
    return df.drop(columns=drop_cols)

application = drop_high_nan_columns(application)
application_test = drop_high_nan_columns(application_test)
bureau = drop_high_nan_columns(bureau)
bureau_balance = drop_high_nan_columns(bureau_balance)
installments = drop_high_nan_columns(installments)
poscash_balance = drop_high_nan_columns(poscash_balance)
creditc_balance = drop_high_nan_columns(creditc_balance)
previous_application = drop_high_nan_columns(previous_application)

# === 3. imputacion básica por mediana y aplicacion de valores nulos === 
def clean_numerical_nans(df):
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    return df

def clean_categorical_nans(df):
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def clean_dataset(df):
    df = clean_numerical_nans(df)
    df = clean_categorical_nans(df)
    return df

# aplicacion de las funciones 
bureau = clean_dataset(bureau)
bureau_balance = clean_dataset(bureau_balance)
installments = clean_dataset(installments)
poscash_balance = clean_dataset(poscash_balance)
creditc_balance = clean_dataset(poscash_balance)
previous_application = clean_dataset(previous_application)

# === 4. Revision de outliers numéricos por rango intercuartil ===
def handle_outliers_iqr(df, strategy='clip', iqr_multiplier=1.5, verbose=True):
    """
    Detecta y maneja outliers en columnas numéricas usando el método IQR.

    Parámetros:
    - df: DataFrame de entrada.
    - strategy: 'clip', 'remove', o 'flag':
        - 'clip': recorta los valores a los límites del IQR.
        - 'remove': elimina las filas que contienen outliers.
        - 'flag': crea columnas indicadoras de si un valor es outlier (no modifica los datos).
    - iqr_multiplier: multiplicador para definir el rango (default 1.5).
    - verbose: si True, imprime resumen de columnas procesadas.

    Retorna:
    - DataFrame modificado según estrategia.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    outlier_info = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = len(outliers)

        if strategy == 'clip':
            df[col] = df[col].clip(lower, upper)
        elif strategy == 'remove':
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif strategy == 'flag':
            df[f'{col}_is_outlier'] = ((df[col] < lower) | (df[col] > upper)).astype(int)

    if verbose:
        print(f"\nOutliers detectados por columna (top 10):")
        print(dict(sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)[:10]))

    return df

# Revision de outliers, se usa la estrategia flag, para evitar dañar el dataset de application
application = handle_outliers_iqr(application, strategy='flag')
application_test = handle_outliers_iqr(application_test, strategy='flag')

# Revision de variables de interés
print(application['DAYS_EMPLOYED'].describe(), application['AMT_REQ_CREDIT_BUREAU_MON'].describe(), application['AMT_REQ_CREDIT_BUREAU_QRT'].describe())

# DAYS_EMPLOYED tiene un comportamiento extraño, la graficamos para ver su comportamiento
application['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram')
plt.xlabel('Days Employment')
plt.show()

# Se trata de un error estructurado-repetitivo, reemplazamos el valor por un nan
application['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
application_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

# Finalmente elimino las columnas de flag, es decir donde se indican los outliers ya que no sigue siendo necesaria
# Eliminar columnas que terminan en "_is_outlier"
application = application.loc[:, ~application.columns.str.endswith('_is_outlier')]
application_test = application_test.loc[:, ~application_test.columns.str.endswith('_is_outlier')]

# === 5. Función de Agregación de Datos Externos ===
def aggregate_numeric(df, group_var, df_name):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df[group_var] = df[group_var]
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    agg.columns = [f'{df_name}_{col[0]}_{col[1]}'.upper() for col in agg.columns]
    return agg.reset_index()

# Primero agregamos bureau_balance por SK_ID_BUREAU
bureau_balance_agg = aggregate_numeric(bureau_balance, 'SK_ID_BUREAU', 'BB')
# Unimos con bureau por SK_ID_BUREAU
bureau = bureau.merge(bureau_balance_agg, on='SK_ID_BUREAU', how='left')

# Agregación de bureau por SK_ID_CURR (ya contiene info de bureau_balance)
bureau_agg = aggregate_numeric(bureau, 'SK_ID_CURR', 'BUREAU')
application = application.merge(bureau_agg, on='SK_ID_CURR', how='left')
application_test = application_test.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Repetir para otras tablas
for df, name in zip(
    [previous_application, installments, creditc_balance, poscash_balance],
    ['PREV', 'INST', 'CC', 'POS']
):
    agg_df = aggregate_numeric(df, 'SK_ID_CURR', name)
    application = application.merge(agg_df, on='SK_ID_CURR', how='left')
    application_test = application_test.merge(agg_df, on='SK_ID_CURR', how='left')

# Eliminar datasets que ya no se van a usar
del bureau, bureau_agg, bureau_balance, bureau_balance_agg, previous_application, installments, creditc_balance, poscash_balance
gc.collect()


# === 6. Análisis descriptivo ===
# Distribución absoluta y relativa de la columna target, es decir la que nos indica si los deudores pagaron o no a tiempo
target_counts = application['TARGET'].value_counts()
target_percent = application['TARGET'].value_counts(normalize=True)

print("Distribución de TARGET:")
print(pd.concat([target_counts, target_percent], axis=1, keys=['Recuento', 'Proporción']))

# Visualización
sns.countplot(x='TARGET', data=application)
plt.title('Distribución de TARGET (0 = Pago a tiempo, 1 = Con Mora)')
plt.xlabel('TARGET')
plt.ylabel('Número de clientes')
plt.show()

# Evaluacion de los desembolsos, los montos financiados y la capacidad de pago.
# Histograma del monto del crédito
sns.histplot(application['AMT_CREDIT'], bins=100, kde=True)
plt.title('Distribución de AMT_CREDIT (Monto del crédito)')
plt.xlabel('Monto del crédito')
plt.show()

# Boxplot por TARGET
sns.boxplot(x='TARGET', y='AMT_CREDIT', data=application)
plt.title('Distribución del monto del crédito por TARGET')
plt.show()

application['AMT_INCOME_TOTAL'].plot.hist(title = 'Income of the client Histogram')
plt.xlabel('Income')
plt.show()

for var in ['AMT_ANNUITY', 'AMT_INCOME_TOTAL']:
    sns.histplot(application[var], bins=100, kde=True)
    plt.title(f'Distribución de {var}')
    plt.show()

# Correlacion de variables numéricas con el comportamiento de pago
# Matriz de correlación
correlations = application.select_dtypes(include=['number']).corr()['TARGET'].sort_values()

print("Variables con correlación negativa más fuerte:")
print(correlations.head(10))

print("\nVariables con correlación positiva más fuerte:")
print(correlations.tail(10)))

# === 7. Codificación de variables categóricas ===
# Copias para no tocar los originales
app_train = application.copy()
app_test = application_test.copy()

# Inicializamos codificador
le = LabelEncoder()

# Aplicamos solo a columnas con exactamente 2 categorías
for col in app_train.columns:
    if app_train[col].dtype == 'object' and app_train[col].nunique() == 2:
        le.fit(app_train[col])
        app_train[col] = le.transform(app_train[col])
        app_test[col] = le.transform(app_test[col])

# Codificación one-hot para variables con mas de 2 categorias
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

# Alineamos los datasets para que tengan las mismas columnas
app_train, app_test = app_train.align(app_test, join='left', axis=1)

print(app_train.shape, app_test.shape)
print(app_train.dtypes.value_counts())

# Imputacion final antes de guardado
# 1. Unir float, int y bool en un único DataFrame numérico
def select_numeric(df):
    return df.select_dtypes(include=['float64', 'int64', 'bool'])

# 2. Imputar ambos datasets
imputer = SimpleImputer(strategy='median')

# Entrenar imputador solo en train
app_train_numeric = select_numeric(app_train)
app_test_numeric = select_numeric(app_test)

imputer.fit(app_train_numeric)

# Imputar
application_train_imputed = pd.DataFrame(
    imputer.transform(app_train_numeric),
    columns=app_train_numeric.columns,
    index=app_train.index
)

application_test_imputed = pd.DataFrame(
    imputer.transform(app_test_numeric),
    columns=app_test_numeric.columns,
    index=app_test.index
)

# 6. Guardar en disco
application_train_imputed.to_csv("app_train_clean.csv", sep=",", index=False)
application_test_imputed.to_csv("app_test_clean.csv", sep=",", index=False)
