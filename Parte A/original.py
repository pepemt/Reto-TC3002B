from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np

# Cargar los datos de testing
path_testing = './Data/data_train.xlsx'
df_testing = pd.read_excel(path_testing)
df_testing = df_testing.dropna(subset=['title', 'text', 'is_suicide'])  # Eliminar filas con valores nulos en 'title' o 'text'
df_testing['title'] = df_testing['title'].astype(str)
df_testing['text'] = df_testing['text'].astype(str)

X_text_testing = df_testing['title'] + ' ' + df_testing['text']
y_testing = df_testing['is_suicide'].map({'no': 0, 'yes': 1})

# Vectorización Bag of Words
vectorizer = CountVectorizer()
X_testing = vectorizer.fit_transform(X_text_testing)

# Configuración de validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=50)

# Modelos a probar
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),  # Habilitar probabilidad para calcular AUC
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    #"LightGBM": lgb.LGBMClassifier()
}

# Testing con validación cruzada
print("=== Testing con Validación Cruzada ===")
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")
    accuracies = []
    f1s = []
    precisions = []
    recalls = []
    aucs = []
    fold = 1

    for train_index, val_index in kf.split(X_testing):
        X_train, X_val = X_testing[train_index], X_testing[val_index]
        y_train, y_val = y_testing.iloc[train_index], y_testing.iloc[val_index]

        # Convertir a float32 si el modelo es LightGBM
        if name == "LightGBM":
            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else np.nan

        accuracies.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        aucs.append(auc)

        print(f"Fold {fold} - Accuracy: {acc:.4f} - F1 Score: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - AUC: {auc:.4f}")
        fold += 1

    print(f"\nResultados finales para {name}:")
    print(f"Promedio Accuracy: {np.mean(accuracies):.4f}")
    print(f"Promedio F1 Score: {np.mean(f1s):.4f}")
    print(f"Promedio Precision: {np.mean(precisions):.4f}")
    print(f"Promedio Recall: {np.mean(recalls):.4f}")
    print(f"Promedio AUC: {np.nanmean(aucs):.4f}")  # Manejar NaN en caso de que AUC no se calcule

# Cargar los datos de validation
path_validation = './Data/data_validation.xlsx'
df_validation = pd.read_excel(path_validation)
df_validation = df_validation.dropna(subset=['title', 'text', 'is_suicide'])  # Eliminar filas con valores nulos en 'title' o 'text'
df_validation['title'] = df_validation['title'].astype(str)
df_validation['text'] = df_validation['text'].astype(str)

X_text_validation = df_validation['title'] + ' ' + df_validation['text']
y_validation = df_validation['is_suicide'].map({'no': 0, 'yes': 1})

# Transformar los datos de validation usando el mismo vectorizador
X_validation = vectorizer.transform(X_text_validation)

# Validación final
print("\n=== Validación Final ===")
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")

    # Convertir a float32 si el modelo es LightGBM
    if name == "LightGBM":
        X_validation = X_validation.astype(np.float32)

    y_pred = model.predict(X_validation)
    y_pred_proba = model.predict_proba(X_validation)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred)
    precision = precision_score(y_validation, y_pred)
    recall = recall_score(y_validation, y_pred)
    auc = roc_auc_score(y_validation, y_pred_proba) if y_pred_proba is not None else np.nan

    print(f"Accuracy: {acc:.4f} - F1 Score: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - AUC: {auc:.4f}")