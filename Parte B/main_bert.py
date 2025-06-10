"""
Script para entrenamiento y evaluación de modelos clásicos usando embeddings de MentalBERT.
Incluye extracción de embeddings, reducción de dimensionalidad, validación cruzada y evaluación final.
Guarda resultados y métricas en archivos Excel.
"""

from Utils.preprocessing_utils import preprocess_text_df_bert
from Utils.models_utils import Models
from sklearn.model_selection import KFold # type: ignore
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix #type: ignore
from transformers import AutoTokenizer, AutoModel   # type: ignore
from sklearn.decomposition import TruncatedSVD # type: ignore
from huggingface_hub import login
import torch # type: ignore
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore
from sklearn.decomposition import PCA # type: ignore

# === Login a Hugging Face Hub ===
login(token='')

# === Carga y preprocesamiento de datos de entrenamiento ===
path_train = './Data/data_train.xlsx'
df_train = pd.read_excel(path_train)
df_train = preprocess_text_df_bert(df_train)  
df_train.to_excel("./Data/data_train_cleaned.xlsx", index=False)  

# Combinar columnas 'title' y 'text' para formar el texto de entrada (ya preprocesado)
X_text_train = df_train['title'] + ' ' + df_train['text']
y_train = df_train['is_suicide'].map({'no': 0, 'yes': 1}).astype(int).to_numpy()

# Nombre del modelo BERT a usar
bert_model = 'sri1208/mental_health_classifier'

all_results = []

def extract_embeddings(text_list, batch_size=32):
    """
    Extrae embeddings de BERT para una lista de textos usando el modelo cargado.
    Procesa los textos en lotes para eficiencia.
    """
    texts = text_list.tolist() if isinstance(text_list, pd.Series) else list(text_list)
    n_samples = len(texts)
    hidden_size = model.config.hidden_size
    embeddings = np.zeros((n_samples, hidden_size), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded = {key: val.to(device) for key, val in encoded.items()}
            outputs = model(**encoded)
            # Usar pooler_output si existe, si no usar CLS
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output
            else:
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings[start:end] = batch_embeddings.cpu().numpy()
    return embeddings

# === Inicialización de modelo y tokenizer de Hugging Face ===
print(f'\n===== Entrenando modelo: {bert_model} =====')
tokenizer = AutoTokenizer.from_pretrained(bert_model)
model = AutoModel.from_pretrained(bert_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# === Extracción de embeddings de entrenamiento ===
X_train_embeddings = extract_embeddings(X_text_train)

pca = PCA(n_components=80)
X_train_reduced = pca.fit_transform(X_train_embeddings)

#lda = LinearDiscriminantAnalysis(n_components=1)  # Solo 1 dimensión para binario
#X_train_reduced = lda.fit_transform(X_train_embeddings, y_train)

# === Validación cruzada con modelos clásicos ===
kf = KFold(n_splits=10, shuffle=True, random_state=50)
models = Models.get_models()

results_cv = []
roc_data = []
conf_matrices = []

print("=== Testing con Validación Cruzada (MentalBERT) ===")
for name, clf in models.items():
    print(f"\n🔍 Modelo: {name}")

    if name == "Deep FFNN":
        X_data = X_train_reduced
        y_data = y_train.astype(np.float32).reshape(-1, 1)
    else:
        X_data = X_train_embeddings
        y_data = y_train

    accuracies, f1s, precisions, recalls, aucs = [], [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_data), start=1):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_pred_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else None
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
        if y_pred_proba is not None:
            for true_label, score in zip(y_val, y_pred_proba):
                roc_data.append({"Modelo": name, "Fold": fold, "y_true": true_label, "y_score": score, "HuggingFaceModel": bert_model})
        # Guardar matriz de confusión del último fold
        if fold == kf.get_n_splits():
            cm = confusion_matrix(y_val, y_pred)
            cm_df = pd.DataFrame(cm, columns=["Pred No", "Pred Yes"], index=["Real No", "Real Yes"])
            cm_df["Modelo"] = name
            cm_df["Fold"] = fold
            cm_df["HuggingFaceModel"] = bert_model
            conf_matrices.append(cm_df.reset_index())
    # Imprimir métricas promedio por modelo
    print(f"\nResultados finales para {name}:")
    print(f"    Promedio Accuracy: {np.mean(accuracies):.4f}")
    print(f"    Promedio F1 Score: {np.mean(f1s):.4f}")
    print(f"    Promedio Precision: {np.mean(precisions):.4f}")
    print(f"    Promedio Recall: {np.mean(recalls):.4f}")
    print(f"    Promedio AUC: {np.nanmean(aucs):.4f}")
    results_cv.append({
        "Modelo": name,
        "Accuracy": np.mean(accuracies),
        "F1 Score": np.mean(f1s),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "AUC": np.nanmean(aucs),
        "HuggingFaceModel": bert_model
    })

# Guardar resultados de validación cruzada y ROC
pd.DataFrame(results_cv).to_excel(f"./Data/resultados_validacion_cruzada_{bert_model.replace('/', '_')}.xlsx", index=False)
pd.DataFrame(roc_data).to_excel(f"./Data/roc_data_{bert_model.replace('/', '_')}.xlsx", index=False)
if conf_matrices:
    pd.concat(conf_matrices).to_excel(f"./Data/matrices_confusion_{bert_model.replace('/', '_')}.xlsx", index=False)
all_results.extend(results_cv)

# Guardar todos los resultados juntos
pd.DataFrame(all_results).to_excel("./Data/resultados_validacion_cruzada_ALL_MODELS.xlsx", index=False)

# === Validación final en datos de test ===
path_test = './Data/data_validation.xlsx'  # Path del conjunto de test/validación final
df_test = pd.read_excel(path_test)
df_test = preprocess_text_df_bert(df_test)  # Preprocesar datos de test de igual forma
X_text_test = df_test['title'] + ' ' + df_test['text']
y_test = df_test['is_suicide'].map({'no': 0, 'yes': 1}).astype(int).to_numpy()

# Extraer embeddings para el set de test
X_test_embeddings = extract_embeddings(X_text_test)

# === REDUCCIÓN DE DIMENSIONALIDAD EN TEST ===
X_test_reduced = pca.transform(X_test_embeddings)   # Si usaste PCA

print("\n=== Validación Final con Embeddings de MentalBERT ===")
results_final = []
for name, model in models.items():

    if name == "Deep FFNN":
        model.fit(X_train_reduced, y_train.astype(np.float32).reshape(-1, 1))
        y_pred = model.predict(X_test_reduced)
        y_pred_proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, "predict_proba") else None
    else:
        model.fit(X_train_embeddings, y_train)
        y_pred = model.predict(X_test_embeddings)
        y_pred_proba = model.predict_proba(X_test_embeddings)[:, 1] if hasattr(model, "predict_proba") else None

    # Calcular métricas en test
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan

    # Imprimir métricas
    print(f"\n🔍 Modelo: {name}")
    print(f"Accuracy: {acc:.4f} - F1: {f1:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - AUC: {auc:.4f}")

    # Guardar métricas en la lista de resultados finales
    results_final.append({
        "Modelo": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc
    })

# Exportar resultados finales a Excel
pd.DataFrame(results_final).to_excel("./Data/resultados_validacion_final.xlsx", index=False)