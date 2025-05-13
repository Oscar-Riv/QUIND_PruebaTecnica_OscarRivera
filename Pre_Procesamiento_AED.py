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

application_train = drop_high_nan(application_train)
application_test = drop_high_nan(application_test)
bureau = drop_high_nan(bureau)
previous_application = drop_high_nan(previous_application)

# === 3. Función de Agregación de Datos Externos ===
def aggregate_numeric(df, group_var, df_name):
    numeric_df = df.select_dtypes(include=['number'])
    numeric_df[group_var] = df[group_var]
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum'])
    agg.columns = [f'{df_name}_{col[0]}_{col[1]}'.upper() for col in agg.columns]
    return agg.reset_index()

# Agregar fuentes externas al conjunto principal
bureau_agg = aggregate_numeric(bureau, 'SK_ID_CURR', 'BUREAU')
installments_agg = aggregate_numeric(installments_payments, 'SK_ID_CURR', 'INST')
previous_agg = aggregate_numeric(previous_application, 'SK_ID_CURR', 'PREV')
poscash_agg = aggregate_numeric(poscash_balance, 'SK_ID_CURR', 'POS')
credit_agg = aggregate_numeric(credit_card_balance, 'SK_ID_CURR', 'CC')

# Unir todos los agregados a application_train y application_test
for agg_df in [bureau_agg, installments_agg, previous_agg, poscash_agg, credit_agg]:
    application_train = application_train.merge(agg_df, on='SK_ID_CURR', how='left')
    application_test = application_test.merge(agg_df, on='SK_ID_CURR', how='left')

# === 4. Codificación de variables categóricas ===
le = LabelEncoder()
for df in [application_train, application_test]:
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() <= 2:
            df[col] = le.fit_transform(df[col].astype(str))

application_train = pd.get_dummies(application_train)
application_test = pd.get_dummies(application_test)

# Alinear columnas de train y test
application_train, application_test = application_train.align(application_test, join='inner', axis=1)

# === 5. Imputación final ===
# Usar mediana para imputar los valores faltantes en numéricos
imputer = SimpleImputer(strategy='median')
application_train = pd.DataFrame(imputer.fit_transform(application_train), columns=application_train.columns)
application_test = pd.DataFrame(imputer.transform(application_test), columns=application_test.columns)

# === 6. Guardado de archivos procesados ===
application_train.to_csv('app_train_clean.csv', index=False)
application_test.to_csv('app_test_clean.csv', index=False)
