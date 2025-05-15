from Utils.preprocessing_utils import preprocess_text_df
from Utils.models_utils import Models
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# === Entrenamiento con preprocesamiento optimizado ===
path_train = './Data/data_train.xlsx'
df_train = pd.read_excel(path_train)
df_train = preprocess_text_df(df_train)
df_train.to_excel("./Data/data_train_cleaned.xlsx", index=False)

X_text_train = df_train['title'] + ' ' + df_train['text']
y_train = df_train['is_suicide'].map({'no': 0, 'yes': 1})

vectorizer = Models.get_models()
smote = SMOTE(random_state=42)

X_train = vectorizer.fit_transform(X_text_train)
X_train, y_train = smote.fit_resample(X_train, y_train)

kf = KFold(n_splits=10, shuffle=True, random_state=50)

models = Models.get_models()

results_cv = []
roc_data = []
conf_matrices = []

print("=== Testing con Validación Cruzada (preprocesamiento optimizado) ===")
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")
    accuracies, f1s, precisions, recalls, aucs = [], [], [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]

        model.fit(X_tr, y_tr)
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

        # Guardar datos para curva ROC
        for true, prob in zip(y_val, y_pred_proba):
            roc_data.append({"Modelo": name, "Fold": fold, "y_true": true, "y_score": prob})

        # Guardar matriz de confusión de último fold
        if fold == kf.get_n_splits():
            cm = confusion_matrix(y_val, y_pred)
            cm_df = pd.DataFrame(cm, columns=["Pred No", "Pred Yes"], index=["Real No", "Real Yes"])
            cm_df['Modelo'] = name
            cm_df['Fold'] = fold
            conf_matrices.append(cm_df.reset_index())

        print(f"Fold {fold} - Accuracy: {acc:.4f} - F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - AUC: {auc:.4f}")

    print(f"\nResultados finales para {name}:")
    print(f"    Promedio Accuracy: {np.mean(accuracies):.4f}")
    print(f"    Promedio F1 Score: {np.mean(f1s):.4f}")
    print(f"    Promedio Precision: {np.mean(precisions):.4f}")
    print(f"    Promedio Recall: {np.mean(recalls):.4f}")
    print(f"    Promedio AUC: {np.nanmean(aucs):.4f}")  # Manejar NaN en caso de que AUC no se calcule
    # Guardar resultados

    results_cv.append({
        "Modelo": name,
        "Accuracy": np.mean(accuracies),
        "F1 Score": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "AUC": np.nanmean(aucs)
    })

# Guardar resultados
pd.DataFrame(results_cv).to_excel("./Data/resultados_validacion_cruzada.xlsx", index=False)
pd.DataFrame(roc_data).to_excel("./Data/roc_data.xlsx", index=False)
pd.concat(conf_matrices).to_excel("./Data/matrices_confusion.xlsx", index=False)

# === Validación final optimizada ===
path_val = './Data/data_validation.xlsx'
df_val = pd.read_excel(path_val)
df_val = preprocess_text_df(df_val)

X_text_val = df_val['title'] + ' ' + df_val['text']
y_val = df_val['is_suicide'].map({'no': 0, 'yes': 1})

X_val = vectorizer.transform(X_text_val)

print("\n=== Validación Final con Datos Preprocesados ===")
results_final = []
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else np.nan

    print(f"Accuracy: {acc:.4f} - F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - AUC: {auc:.4f}")

    results_final.append({
        "Modelo": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc
    })

# Guardar resultados finales
df_results_final = pd.DataFrame(results_final)
df_results_final.to_excel("./Data/resultados_validacion_final.xlsx", index=False)
