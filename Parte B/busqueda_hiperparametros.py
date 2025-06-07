from Utils.preprocessing_utils import preprocess_text_df # type: ignore
from sklearn.model_selection import GridSearchCV, StratifiedKFold # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
import pandas as pd

print("=== Búsqueda de Hiperparámetros para Decision Tree ===")
print("🔍 Cargando y preprocesando datos...")

# === Carga y preprocesamiento de datos ===
path_train = './Data/data_train.xlsx'
df_train = pd.read_excel(path_train)
df_train = preprocess_text_df(df_train)
df_train.to_excel("./Data/data_train_cleaned.xlsx", index=False)

X_text_train = df_train['title'] + ' ' + df_train['text']
y_train = df_train['is_suicide'].map({'no': 0, 'yes': 1})

print("🔠 Vectorizando con TF-IDF...")
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

# === Optimización de Decision Tree ===
param_grid_dt = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}
print("🔍 Buscando mejores parámetros para Decision Tree...")
grid_dt = GridSearchCV(
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    param_grid_dt,
    scoring='accuracy',  # Puedes usar 'f1_macro' o 'f1_weighted'
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_dt.fit(X_train, y_train)
print("\n✅ Mejores parámetros para Decision Tree:", grid_dt.best_params_)
print("🔝 Mejor F1 (macro):", grid_dt.best_score_)

# === Guardar resultados ===
pd.DataFrame(grid_dt.cv_results_).to_excel("./Data/gridsearch_dt_results_f1.xlsx", index=False)

# === Mostrar top resultados ===
print("\n📊 Top combinaciones de Decision Tree:")
print(pd.DataFrame(grid_dt.cv_results_)[['mean_test_score', 'params']].sort_values(by='mean_test_score', ascending=False).head())