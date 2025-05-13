from preprocessing_utils import preprocess_text_df
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

path = './Parte A/data_train.xlsx'
df = pd.read_excel(path)
df = preprocess_text_df(df)

# Saber si hay NaN
print("NaN en el DataFrame:")
print(df.isnull().sum())

df.to_excel("./Parte A/data_train_cleaned.xlsx", index=False)
print("Data cleaned and saved to data_train_cleaned.xlsx")

X_text = df['title'] + ' ' + df['text']  # puedes cambiar esto si prefieres solo 'text'
y = df['is_suicide']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "SVM": SVC(),
    "Naive Bayes": MultinomialNB(),
}

for name, model in models.items():
    print(f"\nModelo: {name}")
    accuracies = []
    fold = 1
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

        print(f"Fold {fold} - Accuracy: {acc:.4f}")
        fold += 1

    print(f"Promedio de Accuracy para {name}: {np.mean(accuracies):.4f}")
