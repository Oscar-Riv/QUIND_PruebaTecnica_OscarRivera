# === 1. Importación de librerías ===
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# === 2. Carga de datos ===
app_train = pd.read_csv("app_train_clean.csv")

# === 3. Reducción de variables ===
# 3.1 Eliminar columnas con baja varianza
vt = VarianceThreshold(threshold=0.01)
vt.fit(app_train)
reduced_vars = app_train.loc[:, vt.get_support()]

# 3.2 Eliminar variables altamente correlacionadas
corr_matrix = reduced_vars.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
filtered = reduced_vars.drop(columns=to_drop)
print(f"Columnas eliminadas por correlación alta: {len(to_drop)}")

# === 4. Preparación para modelado ===
X = filtered.drop(columns='TARGET')
y = filtered['TARGET']

# Escalado de variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Selección de las 20 mejores variables con ANOVA F
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
selected_columns = X.columns[selector.get_support()]
X_selected_df = pd.DataFrame(X_selected, columns=selected_columns)

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_selected_df, y, test_size=0.25, stratify=y, random_state=42
)

# === 5. Modelo LightGBM con RandomizedSearchCV ===
# Balanceo de clases manual
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
lgbm = LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

# Hiperparámetros a explorar
param_dist = {
    'num_leaves': [20, 31, 40, 60, 80],
    'max_depth': [-1, 5, 10, 20, 30],
    'learning_rate': stats.uniform(0.01, 0.2),
    'n_estimators': [100, 200, 300],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': stats.uniform(0.5, 0.5),
    'colsample_bytree': stats.uniform(0.5, 0.5)
}

random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=30,
    scoring='roc_auc',  # puedes cambiar a 'f1' si priorizas balance
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Entrenamiento
random_search.fit(X_train, y_train)
best_lgbm = random_search.best_estimator_

# === 6. Evaluación del modelo ===
y_pred = best_lgbm.predict(X_test)
y_proba = best_lgbm.predict_proba(X_test)[:, 1]

print("\U0001f4ca LightGBM - Clasificación")
print("Mejores hiperparámetros:", random_search.best_params_)
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# === 7. Importancia de variables ===
importances = pd.Series(best_lgbm.feature_importances_, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values[:20], y=importances.index[:20])
plt.title("Importancia de variables - LightGBM")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()
