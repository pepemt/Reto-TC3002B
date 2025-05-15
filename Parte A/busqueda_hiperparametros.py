from preprocessing_utils import preprocess_text_df
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

print("=== Búsqueda de Hiperparámetros para Modelos de Clasificación ===")
print("🔍 Cargando y preprocesando datos...")

# === Carga y preprocesamiento de datos ===
path_train = './Data/data_train.xlsx'
df_train = pd.read_excel(path_train)
df_train = preprocess_text_df(df_train)
df_train.to_excel("./Data/data_train_cleaned.xlsx", index=False)

X_text_train = df_train['title'] + ' ' + df_train['text']
y_train = df_train['is_suicide'].map({'no': 0, 'yes': 1})

print("🔠 Vectorizando con TF-IDF...")
# === Vectorización con TF-IDF mejorado ===
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=2,
    max_features=10000,
    sublinear_tf=True,
    stop_words='english'
)
X_train = vectorizer.fit_transform(X_text_train)

# === Validación cruzada estratificada ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Optimización de Random Forest ===
param_grid_rf = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
}
print("🔍 Buscando mejores parámetros para Random Forest...")
grid_rf = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid_rf,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_rf.fit(X_train, y_train)
print("\n✅ Mejores parámetros para Random Forest:", grid_rf.best_params_)
print("🔝 Mejor F1 (macro):", grid_rf.best_score_)
    
# === Optimización de SVC ===
param_grid_svc = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1, 1, 10]
}
print("\n🔍 Buscando mejores parámetros para SVC...")
grid_svc = GridSearchCV(
    SVC(class_weight='balanced', probability=True, random_state=42),
    param_grid_svc,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_svc.fit(X_train, y_train)
print("\n✅ Mejores parámetros para SVC:", grid_svc.best_params_)
print("🔝 Mejor F1 (macro):", grid_svc.best_score_)

# === Guardar resultados ===
pd.DataFrame(grid_rf.cv_results_).to_excel("./Data/gridsearch_rf_results.xlsx", index=False)
pd.DataFrame(grid_svc.cv_results_).to_excel("./Data/gridsearch_svc_results.xlsx", index=False)

# === Mostrar top resultados ===
print("\n📊 Top combinaciones de Random Forest:")
print(pd.DataFrame(grid_rf.cv_results_)[['mean_test_score', 'params']].sort_values(by='mean_test_score', ascending=False).head())

print("\n📊 Top combinaciones de SVC:")
print(pd.DataFrame(grid_svc.cv_results_)[['mean_test_score', 'params']].sort_values(by='mean_test_score', ascending=False).head())