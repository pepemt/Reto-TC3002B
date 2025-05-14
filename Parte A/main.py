from preprocessing_utils import preprocess_text_df
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# 1. Cargar y preprocesar datos
path = 'Data/data_train.xlsx'
df = pd.read_excel(path)
df = preprocess_text_df(df)
df.to_excel("Data/data_train_cleaned.xlsx", index=False)
print("✅ Data cleaned and saved to data_train_cleaned.xlsx")

# 2. Preparar X e y para el modelo
X_text = df['title'] + ' ' + df['text']
y = df['is_suicide'].map({'no': 0, 'yes': 1})  # Binaria

# 3. Vectorización con TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=2,
    max_features=10000,
    sublinear_tf=True,
    stop_words='english'
)
X = vectorizer.fit_transform(X_text)

# 4. Configurar validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=50)

# 5. Modelos a evaluar
models = {
    "SVM (RBF)": SVC(kernel='rbf', C=1.0, probability=True),
    "SVM (Polynomial)": SVC(kernel='poly', degree=3, C=1.0, probability=True),
    "SVM (Lineal)": SVC(kernel='linear', C=1.0, probability=True),
    "SVM (Sigmoid)": SVC(kernel='sigmoid', C=1.0, probability=True),
    "Naive Bayes": MultinomialNB(alpha=0.5)
}

# 6. Evaluación de cada modelo
for name, model in models.items():
    print(f"\n🔍 Evaluando modelo: {name}")
    accuracies, f1s, aucs = [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        accuracies.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Fold {fold} - Accuracy: {acc:.4f} - F1: {f1:.4f} - AUC: {auc:.4f}")

    print(f"\n📊 Resultados promedio para {name}:")
    print(f"Accuracy: {np.mean(accuracies):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")
    print(f"AUC: {np.mean(aucs):.4f}")
    print()