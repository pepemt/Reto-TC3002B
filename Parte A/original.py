from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Cargar los datos sin preprocesamiento
path = './Parte A/data_train.xlsx'
df = pd.read_excel(path)
# Usar las columnas originales sin preprocesar
X_text = df['title'] + ' ' + df['text']

y = df['is_suicide'].map({'no': 0, 'yes': 1})

# Vectorización TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Configuración de validación cruzada
kf = KFold(n_splits=10, shuffle=True, random_state=50)

# Modelos a probar
models = {
    "SVM (RBF)": SVC(kernel='rbf', C=1.0),
    "SVM (Polynomial)": SVC(kernel='poly', degree=3, C=1.0),
    "SVM (Lineal)": SVC(kernel='linear', C=1.0),
    "SVM (Sigmoid)": SVC(kernel='sigmoid', C=1.0),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=50),
}

# Validación cruzada para cada modelo
for name, model in models.items():
    print(f"\n🔍 Modelo: {name}")
    accuracies = []
    f1s = []
    fold = 1

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        accuracies.append(acc)
        f1s.append(f1)

        print(f"Fold {fold} - Accuracy: {acc:.4f} - F1 Score: {f1:.4f}")
        fold += 1

    print(f"\nResultados finales para {name}:")
    print(f"Promedio Accuracy: {np.mean(accuracies):.4f}")
    print(f"Promedio F1 Score: {np.mean(f1s):.4f}")